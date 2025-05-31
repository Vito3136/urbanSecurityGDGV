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

def executeSVM(goodwares, malwares, is_filtered=False):

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

    '''scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)

    print("Log 4: standardize")

    X_array.clear()'''

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
    print(classification_report(y_test, y_pred, target_names=['Goodware', 'Malware']))