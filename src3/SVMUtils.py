import time, gc, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed

NUM_CORES = 8

# ---------------- dataset.py (o in cima al file) ----------------
class BytecodeDataset(torch.utils.data.Dataset):
    """Converte on-the-fly i campioni in tensori."""
    def __init__(self, X_good, X_mal, indices):
        self.X_good = X_good
        self.X_mal  = X_mal
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        if idx < len(self.X_good):
            feats, label = self.X_good[idx], 0
        else:
            feats, label = self.X_mal[idx - len(self.X_good)], 1
        feats = torch.from_numpy(feats.astype(np.float32))
        label = torch.tensor(label*2 - 1, dtype=torch.float32)
        return feats, label

'''# ---------- Dataset + Dataloader ---------- #
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
                torch.tensor(self.y[idx]))'''

def make_loader(X_good, X_mal, index_array, batch, shuffle, workers):
    ds = BytecodeDataset(X_good, X_mal, index_array)
    return DataLoader(ds,
                      batch_size=batch,
                      shuffle=shuffle,
                      num_workers=workers,
                      pin_memory=True,
                      persistent_workers=(workers > 0))


'''def make_loader(X, y, batch_size, shuffle, num_workers):
    ds = BytecodeDataset(X, y)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)'''

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
    print("1")
    X_good, y_good = load_dataset_from_bytecodes(goodwares, 0, NUM_CORES)
    print("2")
    X_mal , y_mal  = load_dataset_from_bytecodes(malwares, 1, NUM_CORES)
    print("3")

    end = time.time()
    print("4")
    print(f"Log 1: load_dataset_from_bytecodes {(end-start)/60:.2f} min")
    print("5")
    start = time.time()
    print("6")

    # 1) Unisci *solo logicamente* le liste (senza copiarle)
    def idx_to_sample(idx):
        """Dato un indice globale, restituisce (features, label)."""
        if idx < len(X_good):
            return X_good[idx], 0  # goodware
        else:
            return X_mal[idx - len(X_good)], 1  # malware

    print("7")
    N_good, N_mal = len(X_good), len(X_mal)
    print("8")
    N_tot = N_good + N_mal
    print("9")

    # 2) Etichette: vettoriale e leggerissimo
    y = np.concatenate([
        np.zeros(N_good, dtype=np.int8),
        np.ones(N_mal, dtype=np.int8)
    ])
    print("10")

    # 3) Split stratificato sugli INDICI, non sugli oggetti
    all_idx = np.arange(N_tot)
    print("11")
    train_idx, test_idx, y_train, y_test = train_test_split(
        all_idx, y, test_size=0.2, stratify=y, random_state=42)
    print("12")

    end = time.time()
    print("13")
    print("Log 2: train_test_split took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # 5) Loader
    print("14")
    train_loader = DataLoader(BytecodeDataset(train_idx),
                              batch_size=1024, shuffle=True,
                              num_workers=NUM_CORES, pin_memory=True)

    end = time.time()
    print("15")
    print("Log 3: train_loader took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    test_loader = DataLoader(BytecodeDataset(test_idx),
                             batch_size=1024, shuffle=False,
                             num_workers=NUM_CORES, pin_memory=True)

    end = time.time()
    print("16")
    print("Log 4: test_loader took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # ------------------- DEVICE ------------------- #
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    print("17")

    end = time.time()
    print("Log 5: torch.device took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # ------------------- TRAIN -------------------- #
    # input_dim = numero di feature del primo campione del training set
    print("18")
    input_dim = idx_to_sample(train_idx[0])[0].shape[0]

    print("19")
    model = train_svm(device,
                      train_loader,
                      input_dim=input_dim,
                      epochs=epochs,
                      lr=lr)

    print("20")
    end = time.time()
    print("Log 6: train_svm took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # ------------------- PREDICT ------------------ #
    print("21")
    y_pred_t = predict_svm(model, test_loader, device)  # tensor {-1,+1}

    end = time.time()
    print("Log 7: predict_svm took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # ------------------- EVAL --------------------- #
    print("22")
    y_pred = ((y_pred_t.numpy() + 1) // 2).astype(int)  # → {0,1}

    # y_test l’abbiamo già ottenuto da train_test_split ed è {0,1}
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("23")
    print(classification_report(y_test, y_pred,
                                target_names=['Goodware', 'Malware']))

    end = time.time()
    print("Log 8: ultima parte took {:.2f} minutes".format((end - start) / 60))

    '''# Union
    X = np.array(X_good + X_mal, dtype=object)  # dtype=object evita copia inutile
    y = np.array(y_good + y_mal, dtype=np.int8)
    X_good = y_good = X_mal = y_mal = None ; gc.collect()

    # Train / test split
    print("7")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("8")
    end = time.time()
    print("9")
    print("Log 2: train_test_split took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Dataloader
    print("10")
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
                                target_names=['Goodware', 'Malware']))'''


def load_dataset_from_bytecodes(bytecode_list, label, n_jobs=-1):
    # Parallelizza la conversione con joblib
    data = Parallel(n_jobs=n_jobs)(
        delayed(convert_bytecode)(bc) for bc in bytecode_list
    )
    labels = [label] * len(bytecode_list)
    return data, labels

def convert_bytecode(bytecode):
    return np.frombuffer(bytecode, dtype=np.uint8)
