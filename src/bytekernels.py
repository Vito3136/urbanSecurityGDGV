from typing import Union, Callable, Dict, Iterable

BytesLike = Union[bytes, bytearray]

# ---------------------------------------------------------------------------
# 1)  Variable pitch kernel
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
# 2)  Kernel "prime positions": take only bytes in positions with index prime
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
# 3)  Kernel "powers of n": This kernel selects bytes whose positions are successive powers of n within the sequence
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
# 4)  Kernel "checkerboard": divide i dati in blocchi (default da 8 byte). In ogni blocco tiene solo la prima metà
#     Scarta la seconda metà. Il risultato è un flusso “a scacchiera” lineare, metà sì metà no
# ---------------------------------------------------------------------------
def checkerboard_kernel(data: BytesLike, block_size: int = 8) -> bytes:
    return bytes(b for i, b in enumerate(data) if (i % block_size) < block_size // 2)


# ---------------------------------------------------------------------------
# 5)  Kernel "fibonacci filter": tieni i byte in posizioni della sequenza di Fibonacci
#     I primi 25 termini sono: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89,144, 233, 377, 610, 987,1597, 2584, 4181, 6765,10946, 17711, 28657, 46368, 75025
# ---------------------------------------------------------------------------
def fibonacci_index_kernel(data: BytesLike) -> bytes:
    fibs = set()
    a, b = 0, 1
    while a < len(data):
        fibs.add(a)
        a, b = b, a + b
    return bytes(data[i] for i in range(len(data)) if i in fibs)

# ---------------------------------------------------------------------------
# 6)  Kernel "zigzag": Filtra i byte secondo lo schema: on_a sì  •  off_b no  •  off_b sì  •  on_a no  (ripeti)
# ---------------------------------------------------------------------------
def zigzag_kernel(
        data: BytesLike,
        on_a: int = 4,  # primo tratto “tieni”
        off_b: int = 2  # primo tratto “salta”
) -> bytes:
    if on_a <= 0 or off_b <= 0:
        raise ValueError("on_a e off_b devono essere interi positivi (>0)")

    # Costruisce la maschera ciclica 1/0
    pattern = [1] * on_a + [0] * off_b + [1] * off_b + [0] * on_a
    m = len(pattern)

    # Applica il filtro
    return bytes(b for i, b in enumerate(data) if pattern[i % m])


# Estrae solo il byte alla posizione `pos` in ogni blocco di lunghezza `block_size`.
def block_pos_kernel(
        data: BytesLike,
        block_size: int,
        pos: int
) -> bytes:
    if block_size <= 0:
        raise ValueError("block_size deve essere >= 1")

    # Normalizza la posizione come indice positivo all’interno del blocco
    pos = pos % block_size
    if pos < 0 or pos >= block_size:
        raise ValueError("pos deve ricadere nell'intervallo [-(block_size) .. block_size-1]")

    # Se l'inizio (`pos`) cade oltre la lunghezza dei dati, restituiamo bytes vuoto
    if pos >= len(data):
        return b""

    # Costruiamo direttamente la slice con passo `block_size`
    return bytes(data[i] for i in range(pos, len(data), block_size))

# 8. Restituisce i byte i cui indici sono divisibili per `mod`.
def divisible_index_kernel(data: BytesLike, mod: int) -> bytes:
    if mod <= 0:
        raise ValueError("mod deve essere un intero positivo maggiore di 0")

    return bytes(data[i] for i in range(0, len(data)) if i % mod == 0)

# 9. Kernel "spirale compressa": in ogni blocco da 16 prendi solo posizioni 0, 1, 14, 15
def compressed_spiral_kernel(data: BytesLike, block: int = 16) -> bytes:
    positions = {0, 1, block-2, block-1}
    return bytes(data[i] for i in range(len(data)) if (i % block) in positions)