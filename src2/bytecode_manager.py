import os
import shutil
from joblib import Parallel, delayed

NUM_CORES = 8

def collect_bytecodes(root_dir: str, invalid_dir: str) -> list[bytes]:
    """
    Raccoglie file .exe e file senza estensione che iniziano con b'MZ'.
    Restituisce una lista di bytecode binari.
    """
    bytecodes = []
    numberNonValidExe = 0;

    if not os.path.exists(invalid_dir):
        os.makedirs(invalid_dir)

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)

            # Condizione: .exe oppure nessuna estensione
            if fname.lower().endswith(".exe") or '.' not in fname:
                try:
                    with open(full_path, "rb") as f:
                        header = f.read(2)
                        rest = f.read()
                        if header == b'MZ':  # Firma di file PE/EXE
                            bytecodes.append(header + rest)
                        else:
                            print(f"Errore nella lettura di {full_path}, header {header} is not b'MZ'")
                            shutil.move(full_path, os.path.join(invalid_dir, fname))
                            numberNonValidExe = numberNonValidExe + 1
                except Exception as e:
                    print(f"Errore nella lettura di {full_path}: {e}")

    print(f"Non valid data: {numberNonValidExe}")
    return bytecodes

def get_dimension_biggest_bytecode(bytecodes: list[bytes]) -> int:
    # fase 1: calcola tutte le lunghezze in parallelo
    lengths = Parallel(n_jobs=NUM_CORES, backend="loky")(
        delayed(len)(bc) for bc in bytecodes
    )

    # fase 2: restituisce il valore massimo
    return max(lengths)

def pad_bytecode(bytecode, byte_len):
    if len(bytecode) < byte_len:
        return bytecode + b'\x00' * (byte_len - len(bytecode))
    return bytecode