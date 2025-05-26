from typing import Union, Callable, Dict, Iterable
import torch

BytesLike = Union[bytes, bytearray]

# ---------------------------------------------------------------------------
# 1) Variable pitch kernel
# ---------------------------------------------------------------------------
def stride_kernel(data: BytesLike, keep: int, skip: int) -> bytes:
    """
    Returns a new bytes by keeping “keep” bytes and skipping “skip” bytes.
    Example: keep=1, skip=1  -> 1010...
             keep=1, skip=2  -> 100100...
    """
    if keep < 0 or skip < 0:
        raise ValueError("keep and skip must be >= 0")
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        out.extend(data[i : i + keep])
        i += keep + skip
    return bytes(out)

# ---------------------------------------------------------------------------
# 2) Kernel "prime positions": take only bytes in positions with index prime
# ---------------------------------------------------------------------------
def prime_index_kernel(data: BytesLike) -> bytes:
    def is_prime(n: int) -> bool:
        if n < 2: return False # 0 e 1 are NOT prime
        if n == 2: return True # 2 is the first prime number
        if n % 2 == 0: return False # all other peers are NOT first

        # This for looks for any odd divisors from 3 up to the square root of n.
        # If it finds a divisor, n is not prime. If the cycle ends without finding divisors, n is prime.
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0:
                return False
        return True

    # This line creates a new bytes object with only bytes whose indices are prime numbers.
    return bytes(data[i] for i in range(len(data)) if is_prime(i))


# ---------------------------------------------------------------------------
# 3) Kernel "powers of n": This kernel selects bytes whose positions are successive powers of n within the sequence
# ---------------------------------------------------------------------------
def power_of_n_kernel(data: BytesLike, n: int) -> bytes:
    if n <= 1:
        raise ValueError("n must be > 1")
    i = 1
    out = bytearray()
    while i < len(data):
        out.append(data[i])
        i *= n
    return bytes(out)


# ---------------------------------------------------------------------------
# 4) "Checkerboard" Kernel: It divides the data into blocks (by default 8 bytes). 
#    In each block, it keeps only the first half and discards the second half. The result is a linear "checkerboard" stream, alternating half kept, half discarded.
# ---------------------------------------------------------------------------
def checkerboard_kernel(data: BytesLike, block_size: int = 8) -> bytes:
    # Check for the type of data
    if not isinstance(block_size, int) or block_size <= 0:   
        raise ValueError("block_size must be a positive integer greater than 0")
    return bytes(b for i, b in enumerate(data) if (i % block_size) < block_size // 2)


# ---------------------------------------------------------------------------
# 5) "Fibonacci filter" Kernel: Keep the bytes in positions corresponding to the Fibonacci sequence.
#    The first 25 terms are: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025.
# ---------------------------------------------------------------------------
def fibonacci_index_kernel(data: BytesLike) -> bytes:
    # Check for the type of data
    if not hasattr(data, "__getitem__") or not hasattr(data, "__len__"): 
        raise TypeError("data must be a sequence type supporting indexing and length")

    fibs = set()
    a, b = 0, 1
    while a < len(data):
        fibs.add(a)
        a, b = b, a + b
    return bytes(data[i] for i in range(len(data)) if i in fibs)


# ---------------------------------------------------------------------------
# 6) "Zigzag" Kernel: Filters bytes according to the pattern: on_a yes • off_b no • off_b yes • on_a no (repeat)
# ---------------------------------------------------------------------------
def zigzag_kernel(
        data: BytesLike,
        on_a: int = 4,
        off_b: int = 2
) -> bytes:
    if not isinstance(on_a, int) or not isinstance(off_b, int):
        raise TypeError("on_a and off_b have to be integers")
    if on_a <= 0 or off_b <= 0:
        raise ValueError("on_a and off_b have to be positive integers (>0)")
    if not hasattr(data, "__getitem__") or not hasattr(data, "__len__"):
        raise TypeError("data must be a sequence type supporting indexing and length")

    pattern = [1] * on_a + [0] * off_b + [1] * off_b + [0] * on_a
    m = len(pattern)
    return bytes(b for i, b in enumerate(data) if pattern[i % m])


# ---------------------------------------------------------------------------
# 7) Extracts only the byte at position pos in each block of length block_size.
# ---------------------------------------------------------------------------
def block_pos_kernel(
        data: BytesLike,
        block_size: int,
        pos: int
) -> bytes:
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size need to be an integer >= 1")

    if not isinstance(pos, int):
        raise TypeError("pos need to be an intege")

    if not hasattr(data, "__getitem__") or not hasattr(data, "__len__"):
        raise TypeError("data must be an indexable sequence with a defined length")

    # Normalize the position within the block
    pos = pos % block_size

    # If the position exceeds the length of the data, return empty bytes
    if pos >= len(data):
        return b""


    return bytes(data[i] for i in range(pos, len(data), block_size))


# ---------------------------------------------------------------------------
# 8) Returns the bytes whose indices are divisible by mod.
# ---------------------------------------------------------------------------
def divisible_index_kernel(data: BytesLike, mod: int) -> bytes:
    if not isinstance(mod, int):
        raise TypeError("mod need to be an integer")
    if mod <= 0:
        raise ValueError("mod need to be a positive integer (>0)")
    if not hasattr(data, "__getitem__") or not hasattr(data, "__len__"):
        raise TypeError("data must be an indexable sequence with a defined length")

    return bytes(data[i] for i in range(len(data)) if i % mod == 0)


# ---------------------------------------------------------------------------
# 9) "Compressed spiral" Kernel: In each 16-byte block, take only the bytes at positions 0, 1, 14, and 15.
# ---------------------------------------------------------------------------
def compressed_spiral_kernel(data: BytesLike, block: int = 16) -> bytes:
    if not isinstance(block, int):
        raise TypeError("block need to be an integer")
    if block < 4:
        raise ValueError("block need to be an integer >= 4")
    if not hasattr(data, "__getitem__") or not hasattr(data, "__len__"):
        raise TypeError("data must be an indexable sequence with a defined length")

    positions = {0, 1, block - 2, block - 1}
    return bytes(data[i] for i in range(len(data)) if (i % block) in positions)


# ---------------------------------------------------------------------------
# 10) It takes the first and last third of the bytecode, discarding the middle third.
# ---------------------------------------------------------------------------
def tunnel_window_kernel(data: BytesLike) -> bytes:
    if not hasattr(data, "__getitem__") or not hasattr(data, "__len__"):
        raise TypeError("data must be an indexable sequence with a defined length")

    length = len(data)
    if length < 3: #too short to divide in thirds
        return bytes(data)

    one_third = length // 3
    return data[:one_third] + data[-one_third:]

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

class StrideKernel:
    def __init__(self, keep: int, skip: int):
        """
        Filtro A: applica uno stride su una sequenza di byte.
        Ritorna i byte filtrati.
        """
        if keep < 0 or skip < 0:
            raise ValueError("keep and skip must be >= 0")
        self.keep = keep
        self.skip = skip

    def stride_kernel(self, data: bytes) -> bytes:
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            out.extend(data[i : i + self.keep])
            i += self.keep + self.skip
        return bytes(out)

    def __call__(self, bytecode: bytes) -> bytes:
        return self.stride_kernel(bytecode)