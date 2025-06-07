import datetime
import time

import pefile
import math
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim

NUM_CORES = 8

'''def extract_features_dataset_not_filtered(byte_data: bytes):
    try:
        pe = pefile.PE(data=byte_data)
        features = [
            pe.OPTIONAL_HEADER.SizeOfCode,
            pe.OPTIONAL_HEADER.SizeOfInitializedData,
            pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            pe.FILE_HEADER.NumberOfSections,
            pe.FILE_HEADER.TimeDateStamp,
        ]
        return features
    except Exception as e:
        print(f"[!] Errore nell’analisi dei byte: {e}")
        return None


def extract_features_dataset_filterd(
        byte_data: bytes
):
    hist = _byte_histogram(byte_data)
    ent = _entropy(byte_data)
    size = len(byte_data)

    return hist + [ent, size]


# Frequenze dei byte 0-255 (lunghezza fissa 256)
def _byte_histogram(data: bytes) -> list[int]:
    cnt = Counter(data)
    return [cnt.get(i, 0) for i in range(256)]


# Entropia Shannon in bit/byte
def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    cnt = Counter(data)
    total = len(data)
    return -sum((c / total) * math.log2(c / total) for c in cnt.values())

def process_bytecode_filtered(bc, label):
    feats = extract_features_dataset_filterd(bc)
    if feats is not None:
        return feats, label
    else:
        return None

def process_bytecode_not_filterd(bc, label):
    feats = extract_features_dataset_not_filtered(bc)
    if feats is not None:
        return feats, label
    else:
        return None

def load_dataset_from_bytecodes(bytecode_list, label, is_filtered=False):
    data = []
    labels = []
    if is_filtered:
        results = Parallel(n_jobs=NUM_CORES)(
            delayed(process_bytecode_filtered)(bc, label) for bc in bytecode_list
        )
        for res in results:
            if res is not None:
                feats, lbl = res
                data.append(feats)
                labels.append(lbl)
    else:
        results = Parallel(n_jobs=NUM_CORES)(
            delayed(process_bytecode_not_filterd)(bc, label) for bc in bytecode_list
        )
        for res in results:
            if res is not None:
                feats, lbl = res
                data.append(feats)
                labels.append(lbl)

    return data, labels'''

'''def load_dataset_from_bytecodes(bytecode_list, label):
    data = []
    labels = []
    for bytecode in bytecode_list:
        data.append(np.frombuffer(bytecode, dtype=np.uint8))
        labels.append(label)
    return data, labels'''

def convert_bytecode(bytecode):
    return np.frombuffer(bytecode, dtype=np.uint8)

def load_dataset_from_bytecodes(bytecode_list, label, n_jobs=-1):
    # Parallelizza la conversione con joblib
    data = Parallel(n_jobs=n_jobs)(
        delayed(convert_bytecode)(bc) for bc in bytecode_list
    )
    labels = [label] * len(bytecode_list)
    return data, labels

'''def executeSVM(goodwares, malwares, is_filtered=False):

    start = time.time()
    # Loading data
    #X_good, y_good = load_dataset_from_bytecodes(goodwares, 0, is_filtered=is_filtered)
    #X_mal, y_mal = load_dataset_from_bytecodes(malwares, 1, is_filtered=is_filtered)
    X_good, y_good = load_dataset_from_bytecodes(goodwares, 0, NUM_CORES)
    X_mal, y_mal = load_dataset_from_bytecodes(malwares, 1, NUM_CORES)

    end = time.time()
    print("Log 1: load_dataset_from_bytecodes took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Union
    X = np.array(X_good + X_mal)
    y = np.array(y_good + y_mal)

    end = time.time()
    print("Log 2: union took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    X_array = np.stack(X)
    y_array = np.array(y)

    end = time.time()
    print("Log 3: stack e array took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    X = 0
    y = 0

    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X_array)

    #print("Log 4: standardize")

    #X_array.clear()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, stratify=y_array, random_state=42)

    end = time.time()
    print("Log 4: train_test_split took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # SVC model
    model = LinearSVC(class_weight='balanced')
    model.fit(X_train, y_train)

    end = time.time()
    print("Log 5: SVM took {:.2f} minutes".format((end - start) / 60))

    # Assessment¢
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Goodware', 'Malware']))'''

def executeSVM(goodwares, malwares):

    start = time.time()
    # Loading data
    X_good, y_good = load_dataset_from_bytecodes(goodwares, 0, NUM_CORES)
    X_mal, y_mal = load_dataset_from_bytecodes(malwares, 1, NUM_CORES)

    end = time.time()
    print("Log 1: load_dataset_from_bytecodes took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Union
    X = np.array(X_good + X_mal)
    y = np.array(y_good + y_mal)

    end = time.time()
    print("Log 2: union took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    X_array = np.stack(X)
    y_array = np.array(y)

    end = time.time()
    print("Log 3: stack e array took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    X = 0
    y = 0

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, stratify=y_array, random_state=42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device) * 2 - 1  # {-1, +1}
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device) * 2 - 1

    end = time.time()
    print("Log 4: train_test_split took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Addestra il modello
    model = train_svm_from_data(X_train_t, y_train_t)

    X_train_t = 0
    y_train_t = 0

    end = time.time()
    print("Log 5: addestramento modello took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Predici su test set
    y_pred_t = predict_svm(model, X_test_t)

    X_test_t = 0

    end = time.time()
    print("Log 6: predict su test set took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Converte i tensor PyTorch in numpy e da -1/+1 a 0/1
    y_pred = ((y_pred_t.cpu().numpy() + 1) // 2).astype(int)
    y_true = ((y_test_t.cpu().numpy() + 1) // 2).astype(int)

    end = time.time()
    print("Log 7: conversione in numpy took {:.2f} minutes".format((end - start) / 60))
    start = time.time()

    # Valutazione
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['Goodware', 'Malware']))


def train_svm_from_data(X_train, y_train, epochs=10, lr=0.1):
    input_dim = X_train.shape[1]

    class LinearSVM(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        def forward(self, x):
            return self.linear(x)

    def hinge_loss(outputs, labels):
        return torch.mean(torch.clamp(1 - outputs.squeeze() * labels, min=0))

    model = LinearSVM(input_dim).to(X_train.device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = hinge_loss(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return model


def predict_svm(model, X):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        preds = torch.sign(outputs.squeeze())
    return preds
