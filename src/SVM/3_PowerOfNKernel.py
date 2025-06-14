import gc
import sys
import time
from datetime import datetime
from pathlib import Path

from src.SVM.SVMUtils import executeSVM
from src.SVM.bytecode_manager import *
from src.SVM.bytekernels import *

# The `Logger` class redirects the standard output either to the terminal
# or to a log file, ensuring a permanent record of all messages
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Catching global exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__stderr__.write("Manually interrupted.\n")
        return
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

# Current file name
script_name = Path(__file__).stem

# Create folder with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = Path(__file__).parents[2] / "executions_logs" / script_name
log_dir.mkdir(exist_ok=True, parents=True)

# Log file name = script name + .log
log_file = log_dir / f"{timestamp}.log"

logger = Logger(str(log_file))
sys.stdout = logger
sys.stderr = logger

sys.excepthook = handle_exception

NUM_CORES = 8

current_time = datetime.now().strftime("%H:%M:%S")
print("Current time: " + current_time)

base_dir = Path(__file__).parents[2]

# Definition of folder paths
goodwares_path = (base_dir / "resources" / "goodware_dataset").resolve()
non_valid_goodwares_path = (base_dir / "resources" / "non_valid_goodwares").resolve()
malwares_path = (base_dir / "resources" / "malware_dataset").resolve()
non_valid_malwares_path = (base_dir / "resources" / "non_valid_malwares").resolve()

# Creating lists of binary bytecodes
goodware_bytecodes = collect_bytecodes(str(goodwares_path), str(non_valid_goodwares_path))
malware_bytecodes = collect_bytecodes(str(malwares_path), str(non_valid_malwares_path))

# for loop to iterate over the filter parameter
for i in range(2, 6):

    start = time.time()
    powerOfNKernel = PowerOfNKernel(i)

    def filter(b):
        result = powerOfNKernel(b)
        gc.collect()
        return result

    # Filtering goodware bytecodes
    goodware_bytecodes_filtered_with_Power_Of_N_Kernel = Parallel(n_jobs=NUM_CORES)(
        delayed(filter)(b) for b in goodware_bytecodes
    )

    # Filtering malware bytecodes
    malware_bytecodes_filtered_with_Power_Of_N_Kernel = Parallel(n_jobs=NUM_CORES)(
        delayed(filter)(b) for b in malware_bytecodes
    )

    # Calculating the longest length between goodwares and malwares
    lenBiggestGoodware = get_dimension_biggest_bytecode(goodware_bytecodes_filtered_with_Power_Of_N_Kernel)
    lenBiggestMalware = get_dimension_biggest_bytecode(malware_bytecodes_filtered_with_Power_Of_N_Kernel)

    # Zero-padding is performed by creating arrays of length of the largest of malware or goodware + 8
    if (lenBiggestGoodware > lenBiggestMalware):
        lenDef = lenBiggestGoodware + 8
        goodware_bytecodes_filtered_with_Power_Of_N_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in goodware_bytecodes_filtered_with_Power_Of_N_Kernel)
        malware_bytecodes_filtered_with_Power_Of_N_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in malware_bytecodes_filtered_with_Power_Of_N_Kernel)
    else:
        lenDef = lenBiggestMalware + 8
        goodware_bytecodes_filtered_with_Power_Of_N_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in goodware_bytecodes_filtered_with_Power_Of_N_Kernel)
        malware_bytecodes_filtered_with_Power_Of_N_Kernel_zero_padding = Parallel(n_jobs=NUM_CORES)(delayed(pad_bytecode)(b, lenDef) for b in malware_bytecodes_filtered_with_Power_Of_N_Kernel)

    # Cleaning
    goodware_bytecodes_filtered_with_Power_Of_N_Kernel.clear()
    malware_bytecodes_filtered_with_Power_Of_N_Kernel.clear()

    print("Filtered with filter Power of N Kernel with parameter: N = " + str(i))

    # SVM execution
    executeSVM(goodware_bytecodes_filtered_with_Power_Of_N_Kernel_zero_padding, malware_bytecodes_filtered_with_Power_Of_N_Kernel_zero_padding)

    # Cleaning
    goodware_bytecodes_filtered_with_Power_Of_N_Kernel_zero_padding.clear()
    malware_bytecodes_filtered_with_Power_Of_N_Kernel_zero_padding.clear()
    gc.collect()

    # Timestamp
    end = time.time()
    print("Time taken: {:.2f} minutes".format((end - start) / 60))