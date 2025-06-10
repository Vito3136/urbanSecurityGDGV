import time, gc

from src.SVM.SVMUtils import executeSVM
from src.SVM.bytecode_manager import *
from src.SVM.bytekernels import *
from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime

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

start = time.time()
fibonacciIndexKernel = FibonacciIndexKernel()

def filter(b):
    result = fibonacciIndexKernel(b)
    gc.collect()
    return result

# Filtraggio dei bytecodes
goodware_bytecodes_filtered_with_Fibonacci_Index_Kernel = []
goodware_bytecodes_filtered_with_Fibonacci_Index_Kernel = Parallel(n_jobs=NUM_CORES)(
    delayed(filter)(b) for b in goodware_bytecodes
)

print("1")

# Filtraggio dei bytecodes
malware_bytecodes_filtered_with_Fibonacci_Index_Kernel = []
malware_bytecodes_filtered_with_Fibonacci_Index_Kernel = Parallel(n_jobs=NUM_CORES)(
    delayed(filter)(b) for b in malware_bytecodes
)

print("2")

# Calcolata la lunghezza maggiore tra goodwares e malwares
lenBiggestGoodware = get_dimension_biggest_bytecode(goodware_bytecodes_filtered_with_Fibonacci_Index_Kernel)
lenBiggestMalware = get_dimension_biggest_bytecode(malware_bytecodes_filtered_with_Fibonacci_Index_Kernel)

print(lenBiggestGoodware)
print(lenBiggestMalware)

# Si effettua lo zero-padding creando array di lunghezza del piu grande tra malware e goodware + 8
if (lenBiggestGoodware > lenBiggestMalware):
    lenDef = lenBiggestGoodware + 8
    goodware_bytecodes_filtered_with_Fibonacci_Index_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in goodware_bytecodes_filtered_with_Fibonacci_Index_Kernel)
    print("3")
    malware_bytecodes_filtered_with_Fibonacci_Index_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in malware_bytecodes_filtered_with_Fibonacci_Index_Kernel)
    print("4")
else:
    lenDef = lenBiggestMalware + 8
    goodware_bytecodes_filtered_with_Fibonacci_Index_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in goodware_bytecodes_filtered_with_Fibonacci_Index_Kernel)
    print("3")
    malware_bytecodes_filtered_with_Fibonacci_Index_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in malware_bytecodes_filtered_with_Fibonacci_Index_Kernel)
    print("4")

# Pulizia
goodware_bytecodes_filtered_with_Fibonacci_Index_Kernel.clear()
malware_bytecodes_filtered_with_Fibonacci_Index_Kernel.clear()

print("Filtered with filter Prime Index Kernel")
executeSVM(goodware_bytecodes_filtered_with_Fibonacci_Index_zero_padding, malware_bytecodes_filtered_with_Fibonacci_Index_zero_padding)

goodware_bytecodes_filtered_with_Fibonacci_Index_zero_padding.clear()
malware_bytecodes_filtered_with_Fibonacci_Index_zero_padding.clear()
gc.collect()
end = time.time()

print("Time taken: {:.2f} minutes".format((end - start) / 60))