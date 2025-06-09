import time, gc, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed

NUM_CORES = 8

# ---------- Dataset + Dataloader ---------- #
class BytecodeDataset(Dataset):
    """
    Tiene i dati come numpy e converte a tensore solo
    quando il campione serve (→ meno RAM / VRAM).
    """
    def __init__(self, X, y):
        # cast una volta sola, ma resti su numpy
        self.X = X.astype(np.float32)
        self.y = (y.astype(np.float32) * 2 - 1)   # {0,1} → {-1,+1}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # da numpy → torch al volo
        return (torch.from_numpy(self.X[idx]),
                torch.tensor(self.y[idx]))

def make_loader(X, y, batch_size, shuffle, num_workers):
    ds = BytecodeDataset(X, y)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)

# ---------- Modello SVM lineare ---------- #
class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def hinge_loss(outputs, labels):
    return torch.mean(torch.clamp(1 - outputs.squeeze() * labels, min=0))

def train_svm(device, loader, input_dim, epochs=10, lr=0.1):
    model = LinearSVM(input_dim).to(device)
    opt   = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for step, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = hinge_loss(model(xb), yb)
            loss.backward()
            opt.step()

            if (epoch + step / len(loader)) % 0.5 == 0:
                print(f"Epoch {epoch:.1f}, Step {step}, Loss {loss.item():.4f}")
    return model

def predict_svm(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(torch.sign(model(xb).squeeze()).cpu())
    return torch.cat(preds)

# ---------- Pipeline completa ---------- #
def executeSVM(goodwares, malwares, batch_size=1024, epochs=10, lr=0.1):

    start = time.time()
    X_good, y_good = load_dataset_from_bytecodes(goodwares, 0, NUM_CORES)
    X_mal , y_mal  = load_dataset_from_bytecodes(malwares, 1, NUM_CORES)

    print(f"Log 1: load_dataset_from_bytecodes {(time.time()-start)/60:.2f} min")
    start = time.time()

    # Union
    X = np.array(X_good + X_mal, dtype=object)  # dtype=object evita copia inutile
    y = np.array(y_good + y_mal, dtype=np.int8)
    X_good = y_good = X_mal = y_mal = None ; gc.collect()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    end = time.time()
    print("Log 2: train_test_split took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Dataloader
    train_loader = make_loader(X_train, y_train,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=NUM_CORES)

    end = time.time()
    print("Log 3: train_loader took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    test_loader  = make_loader(X_test,  y_test,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=NUM_CORES)

    end = time.time()
    print("Log 4: test_loader took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    end = time.time()
    print("Log 5: torch.device took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Train
    model = train_svm(device, train_loader,
                      input_dim=X_train[0].shape[0],
                      epochs=epochs, lr=lr)

    end = time.time()
    print("Log 6: train_svm took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Predict
    y_pred_t = predict_svm(model, test_loader, device)

    end = time.time()
    print("Log 7: predict_svm took {:.2f} minutes".format((end - start) / 60))

    # Valutazione
    y_pred = ((y_pred_t.numpy() + 1) // 2).astype(int)
    y_true = ((test_loader.dataset.y + 1) // 2).astype(int)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred,
                                target_names=['Goodware', 'Malware']))


def load_dataset_from_bytecodes(bytecode_list, label, n_jobs=-1):
    # Parallelizza la conversione con joblib
    data = Parallel(n_jobs=n_jobs)(
        delayed(convert_bytecode)(bc) for bc in bytecode_list
    )
    labels = [label] * len(bytecode_list)
    return data, labels

def convert_bytecode(bytecode):
    return np.frombuffer(bytecode, dtype=np.uint8)
