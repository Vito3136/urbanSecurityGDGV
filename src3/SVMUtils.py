import time, gc, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed

NUM_CORES = 8

def load_dataset_from_bytecodes(bytecode_list, label, n_jobs=-1):
    # Parallelizza la conversione con joblib
    data = Parallel(n_jobs=n_jobs)(
        delayed(convert_bytecode)(bc) for bc in bytecode_list
    )
    labels = [label] * len(bytecode_list)

    # data = lista di array NumPy (uno per ogni bytecode)
    # labels = lista lunga quanto i campioni, con la stessa etichetta ripetuta
    return data, labels

def convert_bytecode(bytecode):
    # Prende un buffer di byte qualsiasi (bytes, bytearray, memoryview)
    # dtype=np.uint8 dice a NumPy di interpretare ogni singolo byte come un intero senza segno 0-255.
    return np.frombuffer(bytecode, dtype=np.uint8)

# ---------------------------
# 4. Modello linear-SVM
# ---------------------------
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

# ---------------------------
# 6. Predict su nuovi dati
# ---------------------------
def predict(model, device, X_new, batch_size=256):
    """
    X_new: numpy array shape (N, F)
    returns torch.Tensor con valori -1/+1
    """
    loader = DataLoader(TensorDataset(torch.from_numpy(X_new.astype(np.float32))),
                        batch_size=batch_size, shuffle=False, pin_memory=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, in loader:
            xb = xb.to(device)
            preds.append(torch.sign(model(xb)).cpu())
    return torch.cat(preds, dim=0)

def executeSVM(goodwares, malwares, batch_size=1024, epochs=10, lr=0.1):

    # Creazione di lista di n array di interi (byte convertiti) e lista di n labels
    X_good, y_good = load_dataset_from_bytecodes(goodwares, 0, NUM_CORES)
    X_mal , y_mal  = load_dataset_from_bytecodes(malwares, 1, NUM_CORES)

    # ---------------------------
    # 1. Combinazione dei dati
    # ---------------------------
    # X_good, X_mal: lista di ndarray (n_features,)  –> li empiliamo in un'unica matrice
    X = np.vstack(X_good + X_mal).astype(np.float32)  # shape (N, F)
    y_np = np.hstack(y_good + y_mal).astype(np.int64)  # shape (N,)

    # da 0/1 a -1/+1 (lo SVM con hinge loss lo preferisce)
    y_np[y_np == 0] = -1

    # ---------------------------
    # 2. Train / test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_np, test_size=0.2, random_state=42, stratify=y_np)

    # ---------------------------
    # 3. DataLoader (batch al 100 %)
    # ---------------------------
    BATCH_SIZE = 256
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LinearSVM(X.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 = weight_decay
    EPOCHS = 10

    # ---------------------------
    # 5. Training loop
    # ---------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device, dtype=torch.float32)
            optimizer.zero_grad()

            # ---------------------------
            # MIXED-PRECISION QUI 👇
            # ---------------------------
            with torch.autocast(device_type="mps", dtype=torch.float16):
                out = model(xb)
                loss = hinge_loss(out, yb)
            # ---------------------------

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # ––– quick val accuracy –––
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = torch.sign(model(xb)).cpu()  # -1 / +1
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total * 100
        print(f"Epoch {epoch:02d} | loss {epoch_loss:.4f} | val acc {acc:.2f}%")

    y_pred = predict(model, device, X_test)  # -> tensor(-1/+1)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Goodware', 'Malware']))

