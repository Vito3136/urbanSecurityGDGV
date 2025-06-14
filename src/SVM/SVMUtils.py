import copy
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed
import os, shutil, gc

# Number of CPU cores to use for parallel processing
NUM_CORES = 8

def convert_bytecode(bytecode):
    """
    Converts any byte buffer (bytes, bytearray, memoryview) to a NumPy array.
    The dtype=np.uint8 tells NumPy to interpret each byte as an unsigned integer (0-255).
    """
    return np.frombuffer(bytecode, dtype=np.uint8)

class LinearSVM(nn.Module):
    """
    Linear Support Vector Machine implemented as a PyTorch module.
    This is a binary classifier with a single output.
    """
    def __init__(self, n_features):
        """
        Initialize the SVM with a linear layer.
        
        Args:
            n_features: Number of input features
        """
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Output tensor of shape (batch_size,)
        """
        return self.linear(x).squeeze(1)     # (batch,)

def hinge_loss(outputs, targets, C=1.0):
    """
    Computes the hinge loss for SVM training.
    
    Args:
        outputs: Model predictions (any values)
        targets: Target labels in {-1, +1}
        C: Regularization parameter
        
    Returns:
        The mean hinge loss scaled by C
    """
    loss = torch.clamp(1 - targets * outputs, min=0)  # hinge
    return C * loss.mean()

def predict(model, loader, device):
    """
    Makes predictions using the trained model.
    
    Args:
        model: Trained SVM model
        loader: DataLoader containing test samples
        device: Device to run the model on (CPU/GPU)
        
    Returns:
        Tensor of predictions with values in {-1, +1}
    """
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            outs.append(torch.sign(model(xb)).cpu())
    return torch.cat(outs)

def convert_bytecode(bc: bytes | bytearray | memoryview) -> np.ndarray:
    """
    Typed version of convert_bytecode function.
    Converts a byte sequence to a NumPy array of unsigned 8-bit integers.
    
    Args:
        bc: Byte sequence (bytes, bytearray, or memoryview)
        
    Returns:
        NumPy array of unsigned 8-bit integers
    """
    return np.frombuffer(bc, dtype=np.uint8)

def load_dataset_memmap(bytecode_list, label, mmap_dir, fname_prefix,
                        n_jobs=-1, dtype=np.uint8, chunk=256):
    """
    Creates a memory-mapped file on disk and fills it with bytecode data.
    Processes data in chunks to limit RAM usage.
    
    Args:
        bytecode_list: List of bytecode samples
        label: Label to assign to all samples (0 or 1)
        mmap_dir: Directory to store memory-mapped files
        fname_prefix: Prefix for the memory-mapped file name
        n_jobs: Number of parallel jobs (-1 for all cores)
        dtype: Data type for storing the bytecode
        chunk: Number of rows to process simultaneously
        
    Returns:
        Tuple of (X_memmap, y_array) containing features and labels
    """
    os.makedirs(mmap_dir, exist_ok=True)

    N = len(bytecode_list)
    F = len(bytecode_list[0])              # length of one executable
    path = os.path.join(mmap_dir, f"{fname_prefix}.dat")

    X_mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(N, F))
    y      = np.full(N, label, dtype=np.int8)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)

        # Convert max `chunk` executables in parallel
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(convert_bytecode)(bc) for bc in bytecode_list[start:end]
        )

        # Copy each row without creating a giant vstack
        for j, row in enumerate(rows):
            X_mmap[start + j] = row         # memory usage ~1 MB here

        # flush the OS cache every few iterations
        X_mmap.flush()

    return X_mmap, y

def copy_to_memmap(src, dst, dst_start, dst_end, chunk=256, n_jobs=-1):
    """
    
    Args:
        src: Source array
        dst: Destination memmap array
        dst_start: Starting index in destination
        dst_end: Ending index in destination
        chunk: Number of rows to process simultaneously
        n_jobs: Number of parallel jobs
    """
    length = dst_end - dst_start           # number of rows to copy
    for rel_off in range(0, length, chunk):
        rel_stop = min(rel_off + chunk, length)

        # Relative indices for reading from `src`
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(lambda r: r)(src[i])          # convert_bytecode not needed here
            for i in range(rel_off, rel_stop)
        )

        # Absolute indices for writing to `dst`
        abs_off = dst_start + rel_off
        for j, row in enumerate(rows):
            dst[abs_off + j] = row.astype(dst.dtype, copy=False)


def concat_memmaps(a, b, path, dtype=np.float32, chunk=256, n_jobs=-1):
    """
    Concatenates two memory-mapped arrays into a new one.
    
    Args:
        a: First memory-mapped array
        b: Second memory-mapped array
        path: Path for the output memory-mapped file
        dtype: Data type for the output array
        chunk: Number of rows to process simultaneously
        n_jobs: Number of parallel jobs
        
    Returns:
        Concatenated memory-mapped array
    """
    N, F = len(a) + len(b), a.shape[1]
    Xc   = np.memmap(path, mode="w+", dtype=dtype, shape=(N, F))

    # Copy `a` (offset 0 → 0)
    copy_to_memmap(a, Xc, 0, len(a), chunk, n_jobs)

    # Copy `b` (offset len(a) → N)
    copy_to_memmap(b, Xc, len(a), N, chunk, n_jobs)

    Xc.flush()
    return Xc

class MemmapDataset(Dataset):
    """
    PyTorch Dataset implementation for memory-mapped arrays.
    Provides efficient access to memory-mapped data.
    """
    def __init__(self, X_mm, y_arr, idx, dtype=torch.float32):
        """
        Initialize the dataset.
        
        Args:
            X_mm: Memory-mapped feature array
            y_arr: Label array
            idx: Indices to use from the arrays
            dtype: PyTorch data type for tensors
        """
        self.X_mm  = X_mm
        self.y_arr = y_arr
        self.idx   = idx
        self.dtype = dtype

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.idx)

    def __getitem__(self, i):
        """
        Get a sample from the dataset.
        
        Args:
            i: Index of the sample
            
        Returns:
            Tuple of (features, label) as PyTorch tensors
        """
        j   = self.idx[i]  # absolute index in memmap
        # np.asarray => only copy that row and make it C-contiguous
        xnp = np.asarray(self.X_mm[j], dtype=np.float32, order="C")
        ynp = self.y_arr[j].astype(np.float32, copy=False)
        return torch.from_numpy(xnp), torch.tensor(ynp, dtype=torch.float32)


def executeSVM(goodwares, malwares, epochs=50, lr=0.1, patience=5):
    """
    Main function to train and evaluate the SVM model.
    
    Args:
        goodwares: List of goodware bytecodes
        malwares: List of malware bytecodes
        epochs: Maximum number of training epochs
        lr: Learning rate (note: actual implementation uses a different value)
        patience: Number of epochs to wait before early stopping
    """
    # Create lists of integer arrays (converted bytes) and labels
    X_good, y_good = load_dataset_memmap(goodwares, 0, "./mmap", "good", NUM_CORES)
    X_mal, y_mal = load_dataset_memmap(malwares, 1, "./mmap", "mal", NUM_CORES)

    # Combine the data
    X = concat_memmaps(X_good, X_mal, "./mmap/all_float32.dat",
                       dtype=np.float32, chunk=32)
    y_np = np.concatenate((y_good, y_mal), axis=0).astype(np.int64)

    # Convert 0 labels to -1 for SVM training
    y_np[y_np == 0] = -1

    # Train / test split
    indices = np.arange(len(y_np))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y_np
    )

    # Create DataLoaders
    BATCH_SIZE = 256
    train_ds = MemmapDataset(X, y_np, train_idx)
    test_ds = MemmapDataset(X, y_np, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=False, num_workers=0)

    # Model, optimizer, training loop
    # Select best available device (MPS, CUDA, or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = LinearSVM(X.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device)
            optimizer.zero_grad()
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type != "cpu"):
                out = model(xb)
                loss = hinge_loss(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                preds = torch.sign(model(xb)).cpu()
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch:02d} | loss {epoch_loss:.4f} | val acc {acc:.2f}%")

        # Early stopping logic
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stop at epoch {epoch}. Best val acc: {best_acc:.2f}%\n")
            break


    # Final evaluation
    if best_state:
        model.load_state_dict(best_state)

    y_pred = predict(model, test_loader, device).numpy()
    print("Accuracy:", accuracy_score(y_np[test_idx], y_pred))
    print(classification_report(y_np[test_idx], y_pred,
                                target_names=["Goodware", "Malware"]))

    # Cleanup memory and temporary files
    gc.collect()
    shutil.rmtree("./mmap", ignore_errors=True)