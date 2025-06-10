import time, gc

from src3.SVMUtils import executeSVM
from src3.bytecode_manager import *
from src3.bytekernels import *
from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime

NUM_CORES = 8

current_time = datetime.now().strftime("%H:%M:%S")
print("Current time: " + current_time)

base_dir = Path(__file__).parent
goodwares_path = (base_dir.parent / "resources" / "goodware_dataset").resolve()
non_valid_goodwares_path = (base_dir.parent / "resources" / "non_valid_goodwares").resolve()
malwares_path = (base_dir.parent / "resources" / "malware_dataset").resolve()
non_valid_malwares_path = (base_dir.parent / "resources" / "non_valid_malwares").resolve()


# Creazione delle liste di bytecodes binari
goodware_bytecodes = collect_bytecodes(str(goodwares_path), str(non_valid_goodwares_path))
malware_bytecodes = collect_bytecodes(str(malwares_path), str(non_valid_malwares_path))

for i in range(0, 51, 5):
    if (i == 0):
        i = 1
    for j in range(0, 51, 5):
        if (j == 0):
            j = 1

        start = time.time()
        filterStrideKernel = StrideKernel(keep=i, skip=j)

        def filter(b):
            result = filterStrideKernel(b)
            gc.collect()
            return result

        # Filtraggio dei bytecodes
        goodware_bytecodes_filtered_with_Stride_Kernel = []
        goodware_bytecodes_filtered_with_Stride_Kernel = Parallel(n_jobs=NUM_CORES)(
            delayed(filter)(b) for b in goodware_bytecodes
        )

        # Filtraggio dei bytecodes
        malware_bytecodes_filtered_with_Stride_Kernel = []
        malware_bytecodes_filtered_with_Stride_Kernel = Parallel(n_jobs=NUM_CORES)(
            delayed(filter)(b) for b in malware_bytecodes
        )

        # Calcolata la lunghezza maggio tra goodwares e malwares
        lenBiggestGoodware = get_dimension_biggest_bytecode(goodware_bytecodes_filtered_with_Stride_Kernel)
        lenBiggestMalware = get_dimension_biggest_bytecode(malware_bytecodes_filtered_with_Stride_Kernel)

        # Si effettua lo zero-padding creando array di lunghezza del piu grande tra malware e goodware + 8
        if (lenBiggestGoodware > lenBiggestMalware):
            lenDef = lenBiggestGoodware + 8
            goodware_bytecodes_filtered_with_Stride_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in goodware_bytecodes_filtered_with_Stride_Kernel)
            malware_bytecodes_filtered_with_Stride_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in malware_bytecodes_filtered_with_Stride_Kernel)
        else:
            lenDef = lenBiggestMalware + 8
            goodware_bytecodes_filtered_with_Stride_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in goodware_bytecodes_filtered_with_Stride_Kernel)
            malware_bytecodes_filtered_with_Stride_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in malware_bytecodes_filtered_with_Stride_Kernel)

        # Pulizia
        goodware_bytecodes_filtered_with_Stride_Kernel.clear()
        malware_bytecodes_filtered_with_Stride_Kernel.clear()

        print("Filtered with filter Stride Kernel with parameters keep " + str(i) + " skip " + str(j))
        executeSVM(goodware_bytecodes_filtered_with_Stride_Kernel_zero_padding, malware_bytecodes_filtered_with_Stride_Kernel_zero_padding)

        goodware_bytecodes_filtered_with_Stride_Kernel_zero_padding.clear()
        malware_bytecodes_filtered_with_Stride_Kernel_zero_padding.clear()
        gc.collect()
        end = time.time()

        print("Time taken: {:.2f} minutes".format((end - start) / 60))