import time, gc

from typing import List
from torch.utils.data import Dataset, DataLoader
from src3.SVMUtils import executeSVM
from src3.bytecode_manager import *
from src3.bytekernels import *
from joblib import Parallel, delayed

class BytecodeDataset(Dataset):
    """Rende indicizzabile la lista di byte-code per l’uso con DataLoader."""
    def __init__(self, bytecodes: List[bytes]):
        self.bytecodes = bytecodes

    def __len__(self) -> int:
        return len(self.bytecodes)

    def __getitem__(self, idx: int) -> bytes:
        return self.bytecodes[idx]

NUM_CORES = 8
BATCH_SIZE   = 512        # o 1 000, 2 048… regola in base alla RAM
NUM_WORKERS  = 8          # processi di background
pin_memory   = True      # True solo se passi a GPU
PERSISTENT   = True         # mantiene vivi i worker per meno overhead

# Creazione delle liste di bytecodes binari
goodware_bytecodes = collect_bytecodes("/Users/vitoditrani/Desktop/UNIVERSITA/MAGISTRALE/urban_security/urbanSecurityGDGV/resources/goodware_dataset", "/Users/vitoditrani/Desktop/UNIVERSITA/MAGISTRALE/urban_security/urbanSecurityGDGV/resources/non_valid_goodwares")
malware_bytecodes = collect_bytecodes("/Users/vitoditrani/Desktop/UNIVERSITA/MAGISTRALE/urban_security/urbanSecurityGDGV/resources/malware_dataset", "/Users/vitoditrani/Desktop/UNIVERSITA/MAGISTRALE/urban_security/urbanSecurityGDGV/resources/non_valid_malwares")

start = time.time()
primeIndexKernel = PrimeIndexKernel()

'''def filter(b):
    result = primeIndexKernel(b)
    gc.collect()
    return result'''

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ─────────────────────────────────────────────────────────────
# 3. COLLATE FUNCTION
# ─────────────────────────────────────────────────────────────
def collate_and_filter(batch: List[bytes]) -> torch.Tensor:
    """
    1. Esegue primeIndexKernel su CPU per ciascun elemento del batch.
    2. Converte in torch.Tensor.
    3. Sposta sul device (MPS o CPU) in modo non-bloccante.
       - Se *non* ti serve la GPU, puoi restituire direttamente `out_cpu`
         (lista di interi) e saltare completamente la parte tensoriale.
    """
    # Passo 1 – elaborazione CPU
    out_cpu = [primeIndexKernel(b) for b in batch]

    # Passo 2 – tensorizzazione (usa int64 come placeholder; scegli dtype adatto)
    out_tensor = torch.tensor(out_cpu, dtype=torch.int64)

    # Passo 3 – device transfer (solo se MPS disponibile)
    return out_tensor.to(DEVICE, non_blocking=True)

dataset = BytecodeDataset(goodware_bytecodes)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # manteniamo l’ordine originale
    num_workers=NUM_WORKERS,
    collate_fn=collate_and_filter,
    pin_memory=False,       # pinning serve solo su CUDA
    persistent_workers=PERSISTENT,
)


# ---------------- consumo in streaming ----------------
goodware_bytecodes_filtered_with_Prime_Index_Kernel: List[int] = []

for batch_tensor in loader:
    # 👉 Se hai ulteriore logica/ML su GPU, falla qui:
    #     batch_processed = heavy_gpu_model(batch_tensor)
    #     …
    #
    # 👉 Altrimenti, riporta su CPU (se necessario) e converte in lista
    batch_cpu = batch_tensor.cpu().tolist()
    goodware_bytecodes_filtered_with_Prime_Index_Kernel.extend(batch_cpu)

    # Se vuoi consumare pochissima RAM, salva su disco e svuota la lista:
    # save_to_parquet(batch_cpu)
    # del batch_cpu
    gc.collect()

'''# Filtraggio dei bytecodes
goodware_bytecodes_filtered_with_Stride_Kernel = []
goodware_bytecodes_filtered_with_Stride_Kernel = Parallel(n_jobs=NUM_CORES)(
    delayed(filter)(b) for b in goodware_bytecodes
)'''

print("1")

'''# Filtraggio dei bytecodes
malware_bytecodes_filtered_with_Stride_Kernel = []
malware_bytecodes_filtered_with_Stride_Kernel = Parallel(n_jobs=NUM_CORES)(
    delayed(filter)(b) for b in malware_bytecodes
)'''

dataset = BytecodeDataset(malware_bytecodes)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # manteniamo l’ordine originale
    num_workers=NUM_WORKERS,
    collate_fn=collate_and_filter,
    pin_memory=False,       # pinning serve solo su CUDA
    persistent_workers=PERSISTENT,
)


# ---------------- consumo in streaming ----------------
malware_bytecodes_filtered_with_Prime_Index_Kernel: List[int] = []

for batch_tensor in loader:
    # 👉 Se hai ulteriore logica/ML su GPU, falla qui:
    #     batch_processed = heavy_gpu_model(batch_tensor)
    #     …
    #
    # 👉 Altrimenti, riporta su CPU (se necessario) e converte in lista
    batch_cpu = batch_tensor.cpu().tolist()
    malware_bytecodes_filtered_with_Prime_Index_Kernel.extend(batch_cpu)

    # Se vuoi consumare pochissima RAM, salva su disco e svuota la lista:
    # save_to_parquet(batch_cpu)
    # del batch_cpu
    gc.collect()

print("2")

# Calcolata la lunghezza maggio tra goodwares e malwares
lenBiggestGoodware = get_dimension_biggest_bytecode(goodware_bytecodes_filtered_with_Prime_Index_Kernel)
lenBiggestMalware = get_dimension_biggest_bytecode(malware_bytecodes_filtered_with_Prime_Index_Kernel)

print(lenBiggestGoodware)
print(lenBiggestMalware)

# Si effettua lo zero-padding creando array di lunghezza del piu grande tra malware e goodware + 8
if (lenBiggestGoodware > lenBiggestMalware):
    lenDef = lenBiggestGoodware + 8
    goodware_bytecodes_filtered_with_Prime_Index_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in goodware_bytecodes_filtered_with_Prime_Index_Kernel)
    print("3")
    malware_bytecodes_filtered_with_Prime_Index_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in malware_bytecodes_filtered_with_Prime_Index_Kernel)
    print("4")
else:
    lenDef = lenBiggestMalware + 8
    goodware_bytecodes_filtered_with_Prime_Index_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in goodware_bytecodes_filtered_with_Prime_Index_Kernel)
    print("3")
    malware_bytecodes_filtered_with_Prime_Index_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in malware_bytecodes_filtered_with_Prime_Index_Kernel)
    print("4")

# Pulizia
goodware_bytecodes_filtered_with_Prime_Index_Kernel.clear()
malware_bytecodes_filtered_with_Prime_Index_Kernel.clear()

print("Filtered with filter Prime Index Kernel")
executeSVM(goodware_bytecodes_filtered_with_Prime_Index_zero_padding, malware_bytecodes_filtered_with_Prime_Index_zero_padding)

goodware_bytecodes_filtered_with_Prime_Index_zero_padding.clear()
malware_bytecodes_filtered_with_Prime_Index_zero_padding.clear()
gc.collect()
end = time.time()

print("Time taken: {:.2f} minutes".format((end - start) / 60))