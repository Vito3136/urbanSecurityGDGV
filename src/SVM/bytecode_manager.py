import os
import shutil
from joblib import Parallel, delayed

# Define the number of CPU cores to use for parallel processing operations
NUM_CORES = 8

# This module provides utilities for processing executable files (bytecodes)
# Used primarily for malware analysis and classification tasks

def collect_bytecodes(root_dir: str, invalid_dir: str) -> list[bytes]:
    """
    Collects and validates executable files from a directory
    
    This function walks through the specified directory structure to find executable files
    (.exe or files without extension) and validates them by checking for the PE file
    signature (MZ header). Valid files are collected as binary data, while invalid
    files are moved to a separate directory.
    
    Args:
        root_dir: Directory to scan for executable files
        invalid_dir: Directory where invalid files will be moved
        
    Returns:
        A list of binary data (bytes objects) from valid executable files
    """
    bytecodes = []
    numberNonValidExe = 0;

    # Create directory for invalid files if it doesn't exist
    if not os.path.exists(invalid_dir):
        os.makedirs(invalid_dir)

    # Recursively search through all directories and files
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)

            # Process only executable files (.exe) or files without extension
            # Many malware samples don't have proper extensions
            if fname.lower().endswith(".exe") or '.' not in fname:
                try:
                    with open(full_path, "rb") as f:
                        # Check for MZ header (first two bytes of PE files)
                        header = f.read(2)
                        # Read the remaining file content
                        rest = f.read()
                        if header == b'MZ':  # Valid PE file signature
                            # Add the complete file content to our collection
                            bytecodes.append(header + rest)
                        else:
                            # File doesn't have a valid PE header
                            print(f"Error while reading {full_path}, header {header} is not b'MZ'")
                            # Move invalid file to the designated directory
                            shutil.move(full_path, os.path.join(invalid_dir, fname))
                            numberNonValidExe = numberNonValidExe + 1
                except Exception as e:
                    # Handle file access or reading errors
                    print(f"Error while reading {full_path}: {e}")

    # Report statistics on invalid files found
    if numberNonValidExe != 0: 
        print(f"Non valid data: {numberNonValidExe}")
    
    return bytecodes

def get_dimension_biggest_bytecode(bytecodes: list[bytes]) -> int:
    """
    Finds the size of the largest bytecode in the collection
    
    This function uses parallel processing to efficiently calculate
    the length of each binary object in the collection, then returns
    the maximum length found. This is useful for determining padding
    requirements when processing files of varying sizes.
    
    Args:
        bytecodes: A list of binary data from executable files
        
    Returns:
        The length (in bytes) of the largest bytecode in the collection
    """
    # Calculate all lengths in parallel using multiple CPU cores
    # This is significantly faster than sequential processing for large collections
    lengths = Parallel(n_jobs=NUM_CORES, backend="loky")(
        delayed(len)(bc) for bc in bytecodes
    )

    # Find and return the maximum length
    return max(lengths)

def pad_bytecode(bytecode, byte_len):
    """
    Pads a binary object with null bytes to reach a specified length
    
    This function ensures all bytecodes have a consistent length by
    adding null bytes (zeros) to the end of shorter files. This is
    often necessary for machine learning algorithms that require
    fixed-size inputs or for comparative analysis.
    
    Args:
        bytecode: The original binary data to pad
        byte_len: The target length after padding
        
    Returns:
        The padded bytecode if original was shorter than target length,
        otherwise returns the original bytecode unchanged
    """
    if len(bytecode) < byte_len:
        # Add null bytes (zeros) to the end of the bytecode
        return bytecode + b'\x00' * (byte_len - len(bytecode))
    # Return original if already at or exceeding the target length
    return bytecode