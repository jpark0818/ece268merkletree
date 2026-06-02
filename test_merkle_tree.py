import time
import random
import numpy as np
import cupy as cp

from merkle_tree import (
    MerkleTree, 
    BatchMerkleTree, 
    make_poseidon_fn, 
    make_rescue_fn, 
    make_numpy_batch_fn
)
from hashfunctions.RescuePrime import RescuePrime
from hashfunctions import poseidon


def cupy_mod_pow_inplace(b, exp, mod):
    """
    Highly optimized pure CuPy modular exponentiation.
    Uses in-place memory operations (*=, %=) to prevent the GPU 
    from thrashing memory allocations during the while loop.
    """
    result = cp.ones_like(b)
    e = int(exp)  

    while e > 0:
        if e & 1:  
            result *= b
            result %= mod
        b *= b
        b %= mod
        e >>= 1    
        
    return result

def make_fast_pure_cupy_rescue_fn(
    p: int = 4_294_967_291,
    m: int = 3,
    capacity: int = 1,
    security_level: int = 80,
):
    """
    Returns a highly optimized, loop-unrolled CuPy batch hash function.
    No C code, pure Python/CuPy.
    """
    rp = RescuePrime(p=p, m=m, capacity=capacity, security_level=security_level)
    
    p_gpu = cp.uint64(rp.p)
    N_rounds = rp.N

    # Extract MDS matrix into 9 individual scalar constants for loop unrolling
    m00, m01, m02 = cp.uint64(rp.MDS[0][0]), cp.uint64(rp.MDS[0][1]), cp.uint64(rp.MDS[0][2])
    m10, m11, m12 = cp.uint64(rp.MDS[1][0]), cp.uint64(rp.MDS[1][1]), cp.uint64(rp.MDS[1][2])
    m20, m21, m22 = cp.uint64(rp.MDS[2][0]), cp.uint64(rp.MDS[2][1]), cp.uint64(rp.MDS[2][2])

    # Extract Round Constants
    rc_gpu = cp.array(rp.round_constants, dtype=cp.uint64)

    def batch_hash(lefts_gpu, rights_gpu):
        # Structure of Arrays (SoA) - avoids 2D array slicing overhead
        s0 = cp.zeros_like(lefts_gpu)
        s1 = lefts_gpu.copy()
        s2 = rights_gpu.copy()

        for i in range(N_rounds):
            # 1. Forward S-Box
            s0 = (s0 * s0 % p_gpu) * s0 % p_gpu
            s1 = (s1 * s1 % p_gpu) * s1 % p_gpu
            s2 = (s2 * s2 % p_gpu) * s2 % p_gpu
            
            # Extract round constants for this step
            r_base = i * 6
            rc0, rc1, rc2 = rc_gpu[r_base], rc_gpu[r_base+1], rc_gpu[r_base+2]

            # 2 & 3. MDS Matrix + Round Constants (UNROLLED)
            ns0 = (m00*s0 % p_gpu + m01*s1 % p_gpu + m02*s2 % p_gpu + rc0) % p_gpu
            ns1 = (m10*s0 % p_gpu + m11*s1 % p_gpu + m12*s2 % p_gpu + rc1) % p_gpu
            ns2 = (m20*s0 % p_gpu + m21*s1 % p_gpu + m22*s2 % p_gpu + rc2) % p_gpu
            s0, s1, s2 = ns0, ns1, ns2
                
            # 4. Inverse S-Box (Using in-place mod pow)
            s0 = cupy_mod_pow_inplace(s0, rp.alpha_inv, p_gpu)
            s1 = cupy_mod_pow_inplace(s1, rp.alpha_inv, p_gpu)
            s2 = cupy_mod_pow_inplace(s2, rp.alpha_inv, p_gpu)
                
            # Extract round constants for next step
            rc3, rc4, rc5 = rc_gpu[r_base+3], rc_gpu[r_base+4], rc_gpu[r_base+5]

            # 5 & 6. MDS Matrix + Round Constants (UNROLLED)
            ns0 = (m00*s0 % p_gpu + m01*s1 % p_gpu + m02*s2 % p_gpu + rc3) % p_gpu
            ns1 = (m10*s0 % p_gpu + m11*s1 % p_gpu + m12*s2 % p_gpu + rc4) % p_gpu
            ns2 = (m20*s0 % p_gpu + m21*s1 % p_gpu + m22*s2 % p_gpu + rc5) % p_gpu
            s0, s1, s2 = ns0, ns1, ns2
                
        return s1

    return batch_hash


def format_time(seconds: float) -> str:
    """Formats time into a readable string (ms or s)."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.4f} s"

def benchmark_hash_function(name: str, hash_fn, batch_fn, leaf_counts: list, is_gpu: bool = False):
    print(f"\n{'='*75}")
    print(f" Benchmarking: {name}")
    print(f"{'='*75}")
    
    if is_gpu:
        print(f"{'Leaves':<10} | {'Standard CPU Build':<20} | {'GPU CuPy Build':<17} | {'Proof Gen':<12}")
    else:
        print(f"{'Leaves':<10} | {'Standard CPU Build':<20} | {'NumPy CPU Build':<17} | {'Proof Gen':<12}")
    print("-" * 75)

    for n in leaf_counts:
        leaves = [random.randint(1, 10**8) for _ in range(n)]
        
        # CPU Baseline
        start_time = time.perf_counter()
        std_tree = MerkleTree(leaves, hash_fn)
        cpu_build_time = time.perf_counter() - start_time
        
        # Batch Build (NumPy/CuPy)
        if is_gpu:
            leaves_arr = cp.array(leaves, dtype=cp.uint64) 
        else:
            leaves_arr = np.array(leaves, dtype=object)    
            
        start_time = time.perf_counter()
        batch_tree = BatchMerkleTree(leaves_arr, batch_fn)
        
        if is_gpu:
            # Force hardware synchronization before stopping the clock
            cp.cuda.Stream.null.synchronize()
            
        batch_build_time = time.perf_counter() - start_time
        
        # Proof Generation Check
        test_index = random.randint(0, n - 1)
        start_time = time.perf_counter()
        proof = std_tree.get_proof(test_index)
        proof_gen_time = time.perf_counter() - start_time
        
        test_leaf = leaves[test_index]
        is_valid = MerkleTree.verify_proof(test_leaf, proof, std_tree.root, hash_fn)
        assert is_valid, "Proof verification failed!"

        print(f"{n:<10} | {format_time(cpu_build_time):<20} | {format_time(batch_build_time):<17} | {format_time(proof_gen_time):<12}")


def make_fast_pure_cupy_poseidon_fn():
    """
    Returns a highly optimized, loop-unrolled CuPy batch hash function for Poseidon.
    Assumes Poseidon has been scaled to the 31-bit Mersenne Prime.
    """
    from hashfunctions import poseidon # Import Dibyesh's module
    import numpy as np
    
    # Check Dibyesh's initialization flag
    if not poseidon._ready:
        poseidon.poseidon_init()
        
    p_gpu = cp.uint64(poseidon.PRIME)
    R_F = poseidon.R_F
    R_P = poseidon.R_P
    
    # Extract matrices using Dibyesh's exact variable names
    RC = cp.array(poseidon._RC, dtype=cp.uint64)
    MDS = cp.array(poseidon._MDS, dtype=cp.uint64) 
    
    m00, m01, m02 = MDS[0][0], MDS[0][1], MDS[0][2]
    m10, m11, m12 = MDS[1][0], MDS[1][1], MDS[1][2]
    m20, m21, m22 = MDS[2][0], MDS[2][1], MDS[2][2]

    def batch_hash(lefts_gpu, rights_gpu):
        # Structure of Arrays
        s0 = cp.zeros_like(lefts_gpu)  # Capacity element (state[0])
        s1 = lefts_gpu.copy()          # Rate element 1 (state[1])
        s2 = rights_gpu.copy()         # Rate element 2 (state[2])
        
        for r in range(R_F + R_P):
            # 1. Add Round Constants (RC is a 2D array [N_ROUNDS][T])
            s0 = (s0 + RC[r][0]) % p_gpu
            s1 = (s1 + RC[r][1]) % p_gpu
            s2 = (s2 + RC[r][2]) % p_gpu
            
            # 2. S-Boxes (x^5)
            # Full rounds happen at the start and end; Partial rounds in the middle
            is_full = (r < R_F // 2) or (r >= R_F // 2 + R_P)
            
            # Partial round only applies S-box to the capacity element (s0)
            # x^5 = x^2 * x^2 * x
            s0_2 = (s0 * s0) % p_gpu
            s0_4 = (s0_2 * s0_2) % p_gpu
            s0 = (s0_4 * s0) % p_gpu
            
            if is_full:
                s1_2 = (s1 * s1) % p_gpu
                s1_4 = (s1_2 * s1_2) % p_gpu
                s1 = (s1_4 * s1) % p_gpu
                
                s2_2 = (s2 * s2) % p_gpu
                s2_4 = (s2_2 * s2_2) % p_gpu
                s2 = (s2_4 * s2) % p_gpu
                
            # 3. MDS Matrix Multiplication (UNROLLED)
            ns0 = (m00*s0 % p_gpu + m01*s1 % p_gpu + m02*s2 % p_gpu) % p_gpu
            ns1 = (m10*s0 % p_gpu + m11*s1 % p_gpu + m12*s2 % p_gpu) % p_gpu
            ns2 = (m20*s0 % p_gpu + m21*s1 % p_gpu + m22*s2 % p_gpu) % p_gpu
            s0, s1, s2 = ns0, ns1, ns2
            
        return s1 # Squeeze the rate element

    return batch_hash
    
def run_all_benchmarks():
    # Tree sizes up to 8192
    sizes = [16, 256, 2048, 8192, 16384] 
    
    print("Initializing CPU Hash Functions...")
    poseidon_fn = make_poseidon_fn()
    poseidon_batch = make_numpy_batch_fn(poseidon_fn)
    rescue_fn = make_rescue_fn()
    rescue_batch = make_numpy_batch_fn(rescue_fn)
    
    print("Initializing Optimized Pure CuPy Kernel for Rescue-Prime...")
    rescue_gpu_batch = make_fast_pure_cupy_rescue_fn()
    
    # Warm up CuPy context (forces memory pre-allocation)
    _ = rescue_gpu_batch(cp.array([1, 2], dtype=cp.uint64), cp.array([3, 4], dtype=cp.uint64))
    
    print("Initialization Complete. Starting Benchmarks...\n")

    benchmark_hash_function("Poseidon (BN31) - CPU Baseline", poseidon_fn, poseidon_batch, sizes, is_gpu=False)
    benchmark_hash_function("Rescue Prime (32-bit) - CPU Baseline", rescue_fn, rescue_batch, sizes, is_gpu=False)
    benchmark_hash_function("Rescue Prime - OPTIMIZED CUPY ACCELERATION", rescue_fn, rescue_gpu_batch, sizes, is_gpu=True)
    
    print("Initializing Optimized Pure CuPy Kernel for Poseidon...")
    poseidon_gpu_batch = make_fast_pure_cupy_poseidon_fn()
    _ = poseidon_gpu_batch(cp.array([1, 2], dtype=cp.uint64), cp.array([3, 4], dtype=cp.uint64)) # Warmup
    
    benchmark_hash_function("Poseidon - OPTIMIZED CUPY ACCELERATION", poseidon_fn, poseidon_gpu_batch, sizes, is_gpu=True)

if __name__ == "__main__":
    random.seed(42)
    run_all_benchmarks()