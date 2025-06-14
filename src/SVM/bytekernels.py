from typing import Union, ByteString

BytesLike = Union[bytes, bytearray]

"""
        Filter 1: Applies a stride pattern to a byte sequence
        Takes 'keep' bytes, then skips 'skip' bytes, and repeats
        Returns the filtered bytes
"""
class StrideKernel:
    def __init__(self, keep: int, skip: int):

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
    
"""
    Filter 2: Keeps only bytes at indices that are prime numbers
"""
class PrimeIndexKernel:

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def prime_index_kernel(self, data: ByteString) -> bytes:
        return bytes(data[i] for i in range(len(data)) if self._is_prime(i))

    def __call__(self, data: ByteString) -> bytes:
        return self.prime_index_kernel(data)

"""
    Filter 3: Keeps bytes at positions that are consecutive powers of n
    For example, with n=2, keeps bytes at positions 1, 4, 9, 16, 25, etc
"""
class PowerOfNKernel:

    def __init__(self, n: int):
        if n <= 1:
            raise ValueError("n must be > 1")
        self.n = n

    def power_of_n_kernel(self, data: ByteString) -> bytes:
        i = 1
        j = 1
        out = bytearray()
        while i < len(data):
            out.append(data[i])
            j += 1
            i = j**self.n
        return bytes(out)

    def __call__(self, data: ByteString) -> bytes:
        return self.power_of_n_kernel(data)

"""
    Filter 4: Divides the byte stream into blocks of block_size bytes,
    keeping the first half and discarding the second half (linear checkerboard)
"""
class CheckerboardKernel:

    def __init__(self, block_size: int = 8):
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer (>0)")
        self.block_size = block_size

    def checkerboard_kernel(self, data: ByteString) -> bytes:
        bs = self.block_size
        return bytes(b for i, b in enumerate(data) if (i % bs) < bs // 2)

    def __call__(self, data: ByteString) -> bytes:
        return self.checkerboard_kernel(data)

"""
    Filter 5: Keeps bytes at indices that are Fibonacci numbers.
    This creates a non-uniform sampling pattern that follows the
    Fibonacci sequence (0, 1, 1, 2, 3, 5, 8, 13, 21, etc.).
"""
class FibonacciIndexKernel:

    @staticmethod
    def _fibs_upto(n: int) -> set[int]:
        fibs: set[int] = set()
        a, b = 0, 1
        while a < n:
            fibs.add(a)
            a, b = b, a + b
        return fibs

    def fibonacci_index_kernel(self, data: ByteString) -> bytes:
        fibs = self._fibs_upto(len(data))
        return bytes(data[i] for i in range(len(data)) if i in fibs)

    def __call__(self, data: ByteString) -> bytes:
        return self.fibonacci_index_kernel(data)

"""
    Filter 6: Creates a zigzag pattern by keeping ON_A bytes, skipping OFF_B bytes,
    then keeping OFF_B bytes, and skipping ON_A bytes.
    This generates a complex pattern with alternating densities.
"""
class ZigZagKernel:

    def __init__(self, on_a: int = 4, off_b: int = 2):
        if not isinstance(on_a, int) or not isinstance(off_b, int):
            raise TypeError("on_a and off_b must be integers")
        if on_a <= 0 or off_b <= 0:
            raise ValueError("on_a and off_b must be positive (>0)")
        self.on_a = on_a
        self.off_b = off_b
        # Pre-build the pattern for efficiency
        self._pattern = [1] * on_a + [0] * off_b + [1] * off_b + [0] * on_a
        self._m = len(self._pattern)

    def zigzag_kernel(self, data: ByteString) -> bytes:
        m = self._m
        pattern = self._pattern
        return bytes(b for i, b in enumerate(data) if pattern[i % m])

    def __call__(self, data: ByteString) -> bytes:
        return self.zigzag_kernel(data)

"""
    Filter 7: From each block of block_size bytes, extracts only the byte
    at position pos within the block.
"""
class BlockPosKernel:

    def __init__(self, block_size: int, pos: int):
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be an integer ≥ 1")
        if not isinstance(pos, int):
            raise TypeError("pos must be an integer")
        self.block_size = block_size
        self.pos = pos % block_size  # normalize immediately

    def block_pos_kernel(self, data: ByteString) -> bytes:
        if self.pos >= len(data):   # sequence too short
            return b""
        return bytes(data[i] for i in range(self.pos, len(data), self.block_size))

    def __call__(self, data: ByteString) -> bytes:
        return self.block_pos_kernel(data)

"""
    Filter 8: Keeps bytes whose indices are divisible by mod.
"""
class DivisibleIndexKernel:

    def __init__(self, mod: int):
        if not isinstance(mod, int):
            raise TypeError("mod must be an integer")
        if mod <= 0:
            raise ValueError("mod must be > 0")
        self.mod = mod

    def divisible_index_kernel(self, data: ByteString) -> bytes:
        mod = self.mod
        return bytes(data[i] for i in range(len(data)) if i % mod == 0)

    def __call__(self, data: ByteString) -> bytes:
        return self.divisible_index_kernel(data)

"""
    Filter 9: For each block of size 'block', keeps only bytes at positions
    0, 1, block-2, and block-1 (compressed spiral pattern)
"""
class CompressedSpiralKernel:

    def __init__(self, block: int = 16):
        if not isinstance(block, int):
            raise TypeError("block must be an integer")
        if block < 4:
            raise ValueError("block must be ≥ 4")
        self.block = block
        self._positions = {0, 1, block - 2, block - 1}

    def compressed_spiral_kernel(self, data: ByteString) -> bytes:
        blk = self.block
        pos = self._positions
        return bytes(data[i] for i in range(len(data)) if (i % blk) in pos)

    def __call__(self, data: ByteString) -> bytes:
        return self.compressed_spiral_kernel(data)


"""
    Filter 10: Returns the first and last third of the bytecode,
    discarding the middle third (tunnel window pattern)
"""
class TunnelWindowKernel:

    @staticmethod
    def tunnel_window_kernel(data: ByteString) -> bytes:
        length = len(data)
        if length < 3:
            return bytes(data)
        one_third = length // 3
        return data[:one_third] + data[-one_third:]

    def __call__(self, data: ByteString) -> bytes:
        return self.tunnel_window_kernel(data)

"""
    Filter 11: Returns only the middle third of the bytecode,
    discarding the first and last thirds (inverse tunnel window)
"""
class ReverseTunnelWindowKernel:

    @staticmethod
    def tunnel_window_kernel(data: ByteString) -> bytes:
        length = len(data)
        if length < 3:
            return bytes(data)
        one_third = length // 3
        return data[one_third:2 * one_third]

    def __call__(self, data: ByteString) -> bytes:
        return self.tunnel_window_kernel(data)