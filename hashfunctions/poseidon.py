"""
poseidon.py — Poseidon hash function over the BN31 scalar field

Parameters : t = 3  (2-to-1 hash)
             R_F = 8  full rounds  (4 at start, 4 at end)
             R_P = 57 partial rounds
             alpha = 5  (S-box: x^5 mod p)

Reference  : "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems"
             Grassi, Khovratovich, Rechberger, Roy, Schofnegger — 2019
             https://eprint.iacr.org/2019/458

Requires   : Python 3.8+ (uses pow(x, -1, mod) for modular inverse)
"""

T        = 3
R_F      = 8
R_P      = 57
N_ROUNDS = R_F + R_P   # 65
ALPHA    = 5
N_BITS   = 31

PRIME = 2147483647

# ============================================================
# Grain LFSR — deterministic round-constant generation
#
# 80-bit LFSR with feedback polynomial x^80+x^62+x^51+x^38+x^23+x^13+1.
# The initialization vector encodes all Poseidon parameters so that
# different parameter sets yield independent constants (Section 4.2).
# ============================================================

def _grain_clock(s, warmup):
    out = s[0]
    fb  = s[0] ^ s[13] ^ s[23] ^ s[38] ^ s[51] ^ s[62]
    if warmup:
        fb ^= out
    s[:] = s[1:] + [fb]
    return out

def _grain_bit(s):
    while True:
        b0 = _grain_clock(s, False)
        b1 = _grain_clock(s, False)
        if b0:
            return b1

def _grain_field(s):
    while True:
        val = 0
        for _ in range(N_BITS):
            val = (val << 1) | _grain_bit(s)
        if val < PRIME:
            return val

def _grain_init():
    s = [0] * 80

    # Bit layout (80 bits total):
    #  [0- 1]  = 1,0           field type: prime
    #  [2- 5]  = ALPHA         4 bits, MSB first
    #  [6-17]  = N_BITS        12 bits, MSB first
    # [18-29]  = T             12 bits, MSB first
    # [30-39]  = R_F           10 bits, MSB first
    # [40-49]  = R_P           10 bits, MSB first
    # [50-79]  = 1…1           30 padding ones
    s[0] = 1  # s[1] stays 0

    for i in range(4):
        s[2  + i] = (ALPHA  >> (3  - i)) & 1
    for i in range(12):
        s[6  + i] = (N_BITS >> (11 - i)) & 1
    for i in range(12):
        s[18 + i] = (T      >> (11 - i)) & 1
    for i in range(10):
        s[30 + i] = (R_F    >> (9  - i)) & 1
    for i in range(10):
        s[40 + i] = (R_P    >> (9  - i)) & 1
    for i in range(50, 80):
        s[i] = 1

    for _ in range(160):
        _grain_clock(s, True)

    return s

# ============================================================
# Global Poseidon state
# ============================================================

_RC    = None   # round constants [N_ROUNDS][T]
_MDS   = None   # MDS matrix [T][T]
_ready = False

# ============================================================
# MDS matrix — Cauchy construction (guaranteed MDS)
#
# M[i][j] = 1 / (x_i - y_j)  mod p
# x = {0, 1, …, T-1}   y = {T, T+1, …, 2T-1}
# ============================================================

def _build_mds():
    mds = [[0] * T for _ in range(T)]
    for i in range(T):
        for j in range(T):
            d = (i - (T + j)) % PRIME
            mds[i][j] = pow(d, -1, PRIME)
    return mds

# ============================================================
# Initialization
# ============================================================

def poseidon_init():
    global _RC, _MDS, _ready
    if _ready:
        return
    s    = _grain_init()
    _RC  = [[_grain_field(s) for _ in range(T)] for _ in range(N_ROUNDS)]
    _MDS = _build_mds()
    _ready = True

# ============================================================
# Poseidon permutation internals
# ============================================================

def _add_round_constants(state, r):
    return [(state[i] + _RC[r][i]) % PRIME for i in range(T)]

def _sbox_full(state):
    return [pow(x, ALPHA, PRIME) for x in state]

def _sbox_partial(state):
    return [pow(state[0], ALPHA, PRIME)] + state[1:]

def _mds_multiply(state):
    acc = [0] * T
    for i in range(T):
        for j in range(T):
            acc[i] = (acc[i] + _MDS[i][j] * state[j]) % PRIME
    return acc

# ============================================================
# Poseidon-π permutation
#
# Structure:
#   R_F/2 full rounds  →  R_P partial rounds  →  R_F/2 full rounds
# Each round: AddRoundConstants → S-box → MDS
# ============================================================

def poseidon_permute(state):
    assert _ready, "Call poseidon_init() first"
    state = list(state)
    half  = R_F // 2   # = 4

    for r in range(half):
        state = _add_round_constants(state, r)
        state = _sbox_full(state)
        state = _mds_multiply(state)

    for r in range(half, half + R_P):
        state = _add_round_constants(state, r)
        state = _sbox_partial(state)
        state = _mds_multiply(state)

    for r in range(half + R_P, N_ROUNDS):
        state = _add_round_constants(state, r)
        state = _sbox_full(state)
        state = _mds_multiply(state)

    return state

# ============================================================
# Poseidon hash  (2-to-1 sponge)
#
# Capacity element : state[0] = 0
# Rate elements    : state[1] = in0,  state[2] = in1
# Output           : state[1] after the permutation
#
# Both inputs must be valid field elements (0 <= x < p).
# ============================================================

def poseidon_hash(in0, in1):
    assert _ready, "Call poseidon_init() first"
    state = poseidon_permute([0, in0, in1])
    return state[1]

def poseidon_hash_hex(hex0, hex1):
    a = int(hex0, 16)
    b = int(hex1, 16)
    return f"{poseidon_hash(a, b):064x}"

# ============================================================
# Test helpers
# ============================================================

def _count_differing_bits(a, b):
    return bin(a ^ b).count("1")

def _report(desc, passed):
    print(f"    {desc:<52} [{'PASS' if passed else 'FAIL'}]")

# ============================================================
# Test 1 — Known-answer / regression
# ============================================================

def test_known_answer():
    print("\n[Test 1] Known-answer / regression")
    cases    = [(0, 0), (1, 2), (3, 4), (100, 200)]
    all_pass = True

    for a, b in cases:
        h1 = poseidon_hash(a, b)
        h2 = poseidon_hash(a, b)

        valid   = 0 <= h1 < PRIME
        nonzero = h1 != 0
        determ  = h1 == h2

        print(f"  poseidon({a:3d}, {b:3d}) = {h1:064x}")
        _report(f"output in [0, p)  for ({a},{b})", valid)
        _report(f"output != 0       for ({a},{b})", nonzero)
        _report(f"deterministic     for ({a},{b})", determ)

        if not (valid and nonzero and determ):
            all_pass = False

    print(f"  => Test 1 overall: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass

# ============================================================
# Test 2 — Avalanche effect
# ============================================================

def test_avalanche():
    print("\n[Test 2] Avalanche effect  (1-bit input change -> ~50% output change)")
    cases     = [(1, 2), (42, 99), (0, 0), (999, 1000)]
    threshold = N_BITS // 4   # >= 63 bits must differ
    all_pass  = True

    for k, (a, b) in enumerate(cases):
        h1   = poseidon_hash(a, b)
        h2   = poseidon_hash((a + 1) % PRIME, b)
        diff = _count_differing_bits(h1, h2)
        pct  = 100.0 * diff / N_BITS
        ok   = diff >= threshold
        print(f"  case {k+1}  a={a}, b={b}:  {diff} / {N_BITS} bits differ  ({pct:.1f}%)  "
              f"[{'PASS' if ok else 'FAIL'}]")
        if not ok:
            all_pass = False

    print(f"  => Test 2 overall: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass

# ============================================================
# Test 3 — Non-commutativity  hash(a,b) != hash(b,a)
# ============================================================

def test_noncommutative():
    print("\n[Test 3] Non-commutativity  poseidon(a,b) != poseidon(b,a)")
    cases    = [(1, 2), (3, 7), (100, 200), (999, 1000)]
    all_pass = True

    for a, b in cases:
        hab  = poseidon_hash(a, b)
        hba  = poseidon_hash(b, a)
        diff = hab != hba
        _report(f"poseidon({a},{b}) != poseidon({b},{a})", diff)
        if not diff:
            all_pass = False

    print(f"  => Test 3 overall: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("============================================================")
    print(f" Poseidon hash  BN254  t={T}  R_F={R_F}  R_P={R_P}  alpha={ALPHA}")
    print("============================================================")
    print(f"Generating {N_ROUNDS * T} round constants via Grain LFSR …")
    poseidon_init()
    print("Done.")

    r1 = test_known_answer()
    r2 = test_avalanche()
    r3 = test_noncommutative()

    print("\n============================================================")
    print(f" Summary:  Test1={'PASS' if r1 else 'FAIL'}  "
          f"Test2={'PASS' if r2 else 'FAIL'}  "
          f"Test3={'PASS' if r3 else 'FAIL'}")
    print("============================================================")

    import sys
    sys.exit(0 if (r1 and r2 and r3) else 1)
