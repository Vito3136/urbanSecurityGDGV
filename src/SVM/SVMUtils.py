import copy
import time, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed
import os, shutil, gc

NUM_CORES = 8

def convert_bytecode(bytecode):
    # Prende un buffer di byte qualsiasi (bytes, bytearray, memoryview)
    # dtype=np.uint8 dice a NumPy di interpretare ogni singolo byte come un intero senza segno 0-255.
    return np.frombuffer(bytecode, dtype=np.uint8)

class LinearSVM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(1)     # (batch,)

def hinge_loss(outputs, targets, C=1.0):
    """targets in {-1, +1}; outputs qualunque"""
    loss = torch.clamp(1 - targets * outputs, min=0)  # hinge
    return C * loss.mean()

def predict(model, loader, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            outs.append(torch.sign(model(xb)).cpu())
    return torch.cat(outs)

def convert_bytecode(bc: bytes | bytearray | memoryview) -> np.ndarray:
    return np.frombuffer(bc, dtype=np.uint8)

def load_dataset_memmap(bytecode_list, label, mmap_dir, fname_prefix,
                        n_jobs=-1, dtype=np.uint8, chunk=256):
    """
    Crea un file .dat su disco e lo riempie senza mai superare
    `chunk` righe contemporaneamente in RAM.
    Restituisce (X_memmap, y_array).
    """
    os.makedirs(mmap_dir, exist_ok=True)

    N = len(bytecode_list)
    F = len(bytecode_list[0])              # lunghezza di un eseguibile
    path = os.path.join(mmap_dir, f"{fname_prefix}.dat")

    X_mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(N, F))
    y      = np.full(N, label, dtype=np.int8)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)

        # Converte max `chunk` eseguibili in parallelo
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(convert_bytecode)(bc) for bc in bytecode_list[start:end]
        )

        # Copia ogni riga senza fare un vstack gigantesco
        for j, row in enumerate(rows):
            X_mmap[start + j] = row         # qui memoria ~1 MB

        # Facoltativo: svuota la cache OS ogni tot iterazioni
        X_mmap.flush()

    return X_mmap, y

def copy_to_memmap(src, dst, dst_start, dst_end, chunk=256, n_jobs=-1):
    """
    Copia src[0:dst_end-dst_start] → dst[dst_start:dst_end]
    (quindi `dst_end - dst_start == len(src)` quando copiamo b)
    """
    length = dst_end - dst_start           # quante righe devo copiare
    for rel_off in range(0, length, chunk):
        rel_stop = min(rel_off + chunk, length)

        # indici relativi per leggere da `src`
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(lambda r: r)(src[i])          # qui non serviva convert_bytecode
            for i in range(rel_off, rel_stop)
        )

        # indici assoluti per scrivere in `dst`
        abs_off = dst_start + rel_off
        for j, row in enumerate(rows):
            dst[abs_off + j] = row.astype(dst.dtype, copy=False)


def concat_memmaps(a, b, path, dtype=np.float32, chunk=256, n_jobs=-1):
    N, F = len(a) + len(b), a.shape[1]
    Xc   = np.memmap(path, mode="w+", dtype=dtype, shape=(N, F))

    # copia `a` (offset 0 → 0)
    copy_to_memmap(a, Xc, 0, len(a), chunk, n_jobs)

    # copia `b` (offset len(a) → N)
    copy_to_memmap(b, Xc, len(a), N, chunk, n_jobs)

    Xc.flush()
    return Xc

class MemmapDataset(Dataset):
    def __init__(self, X_mm, y_arr, idx, dtype=torch.float32):
        self.X_mm  = X_mm
        self.y_arr = y_arr
        self.idx   = idx
        self.dtype = dtype

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j   = self.idx[i]                     # indice assoluto nel memmap
        # np.asarray => copia solo quella riga e la rende C-contigua
        xnp = np.asarray(self.X_mm[j], dtype=np.float32, order="C")
        ynp = self.y_arr[j].astype(np.float32, copy=False)
        return torch.from_numpy(xnp), torch.tensor(ynp, dtype=torch.float32)


def executeSVM(goodwares, malwares, epochs=50, lr=0.1, patience=5):

    # Creazione di lista di n array di interi (byte convertiti) e lista di n labels
    #X_good, y_good = load_dataset_from_bytecodes(goodwares, 0, NUM_CORES)
    #X_mal , y_mal  = load_dataset_from_bytecodes(malwares, 1, NUM_CORES)

    X_good, y_good = load_dataset_memmap(goodwares, 0, "./mmap", "good", NUM_CORES)
    X_mal, y_mal = load_dataset_memmap(malwares, 1, "./mmap", "mal", NUM_CORES)

    # ---------------------------
    # 1. Combinazione dei dati
    # ---------------------------
    # X_good, X_mal: lista di ndarray (n_features,)  –> li empiliamo in un'unica matrice
    #X = np.vstack(X_good + X_mal).astype(np.float32)  # shape (N, F)
    #X = np.concatenate((X_good, X_mal), axis=0).astype(np.float32)
    X = concat_memmaps(X_good, X_mal, "./mmap/all_float32.dat",
                       dtype=np.float32, chunk=32)
    #y_np = np.hstack(y_good + y_mal).astype(np.int64)  # shape (N,)
    y_np = np.concatenate((y_good, y_mal), axis=0).astype(np.int64)

    # da 0/1 a -1/+1 (lo SVM con hinge loss lo preferisce)
    #y_np[y_np == 0] = -1
    y_np[y_np == 0] = -1

    # ---------------------------
    # 2. Train / test split
    # ---------------------------
    #X_train, X_test, y_train, y_test = train_test_split(X, y_np, test_size=0.2, random_state=42, stratify=y_np)

    indices = np.arange(len(y_np))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y_np
    )

    # ---------------------------
    # 4. DataLoader
    # ---------------------------
    BATCH_SIZE = 256
    train_ds = MemmapDataset(X, y_np, train_idx)
    test_ds = MemmapDataset(X, y_np, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=False, num_workers=0)

    # ---------------------------
    # 5. Modello, ottimizzatore, ciclo di training
    # ---------------------------
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = LinearSVM(X.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ----- train -----
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

        # ----- validation -----
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

        # early stopping
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stop at epoch {epoch}. Best val acc: {best_acc:.2f}%\n")
            break

    # ---------------------------
    # 6. Test finale
    # ---------------------------
    if best_state:
        model.load_state_dict(best_state)

    y_pred = predict(model, test_loader, device).numpy()
    print("Accuracy:", accuracy_score(y_np[test_idx], y_pred))
    print(classification_report(y_np[test_idx], y_pred,
                                target_names=["Goodware", "Malware"]))

    gc.collect()
    shutil.rmtree("./mmap", ignore_errors=True)

