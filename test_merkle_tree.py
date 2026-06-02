import time
import random
import numpy as np
import cupy as cp

# Import CPU modules from your merkle_tree.py
from merkle_tree import (
    MerkleTree, 
    BatchMerkleTree, 
    make_poseidon_fn, 
    make_rescue_fn, 
    make_numpy_batch_fn
)
from hashfunctions.RescuePrime import RescuePrime



def cupy_mod_pow(base_array, exp, mod):
    """Computes (base_array ^ exp) % mod for CuPy arrays."""
    result = cp.ones_like(base_array, dtype=cp.uint64)
    b = base_array % mod
    e = int(exp)  

    while e > 0:
        if e & 1:  
            result = (result * b) % mod
        b = (b * b) % mod
        e >>= 1    
        
    return result

def make_pure_cupy_batch_rescue_fn(
    p: int = 4_294_967_291,
    m: int = 3,
    capacity: int = 1,
    security_level: int = 80,
):
    """Returns a CuPy-vectorised batch hash function for BatchMerkleTree."""
    rp = RescuePrime(p=p, m=m, capacity=capacity, security_level=security_level)
    
    p_gpu = cp.uint64(rp.p)
    MDS_gpu = cp.array(rp.MDS, dtype=cp.uint64)
    rc_gpu = cp.array(rp.round_constants, dtype=cp.uint64)
    N_rounds = rp.N

    def cupy_mat_vec_mul(matrix, state_batch):
        new_state = cp.zeros_like(state_batch)
        for i in range(m):
            acc = cp.zeros(state_batch.shape[1], dtype=cp.uint64)
            for j in range(m):
                term = (matrix[i][j] * state_batch[j]) % p_gpu
                acc = (acc + term) % p_gpu
            new_state[i] = acc
        return new_state

    def batch_hash(lefts_gpu, rights_gpu):
        batch_size = lefts_gpu.shape[0]
        state = cp.zeros((m, batch_size), dtype=cp.uint64)
        state[1, :] = lefts_gpu
        state[2, :] = rights_gpu

        for i in range(N_rounds):
            state = (state * state % p_gpu) * state % p_gpu
            state = cupy_mat_vec_mul(MDS_gpu, state)
            
            rc_step1 = rc_gpu[i * 2 * m : i * 2 * m + m].reshape(m, 1)
            state = (state + rc_step1) % p_gpu
                
            state = cupy_mod_pow(state, rp.alpha_inv, p_gpu)
            state = cupy_mat_vec_mul(MDS_gpu, state)
            
            rc_step2 = rc_gpu[i * 2 * m + m : i * 2 * m + 2 * m].reshape(m, 1)
            state = (state + rc_step2) % p_gpu
                
        return state[1, :]

    return batch_hash



def format_time(seconds: float) -> str:
    """Formats time into a readable string (ms or s)."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.4f} s"

def benchmark_hash_function(name: str, hash_fn, batch_fn, leaf_counts: list, is_gpu: bool = False):
    """Benchmarks tree construction, proof generation, and verification."""
    print(f"\n{'='*75}")
    print(f" Benchmarking: {name}")
    print(f"{'='*75}")
    
    if is_gpu:
        print(f"{'Leaves':<10} | {'Standard CPU Build':<20} | {'GPU CuPy Build':<17} | {'Proof Gen':<12}")
    else:
        print(f"{'Leaves':<10} | {'Standard CPU Build':<20} | {'NumPy CPU Build':<17} | {'Proof Gen':<12}")
    print("-" * 75)

    for n in leaf_counts:
        # 1. Generate Leaves
        leaves = [random.randint(1, 10**8) for _ in range(n)]
        
        # 2. Standard CPU Tree Build (Baseline)
        start_time = time.perf_counter()
        std_tree = MerkleTree(leaves, hash_fn)
        cpu_build_time = time.perf_counter() - start_time
        
        # 3. Batch Tree Build (NumPy or CuPy)
        if is_gpu:
            leaves_arr = cp.array(leaves, dtype=cp.uint64) # Send to GPU memory
        else:
            leaves_arr = np.array(leaves, dtype=object)    # Keep in CPU memory
            
        start_time = time.perf_counter()
        batch_tree = BatchMerkleTree(leaves_arr, batch_fn)
        
        # Force CuPy synchronization to ensure accurate timing
        if is_gpu:
            cp.cuda.Stream.null.synchronize()
            
        batch_build_time = time.perf_counter() - start_time
        
        # 4. Proof Generation (Single Leaf via CPU)
        test_index = random.randint(0, n - 1)
        start_time = time.perf_counter()
        proof = std_tree.get_proof(test_index)
        proof_gen_time = time.perf_counter() - start_time
        
        # 5. Proof Verification
        test_leaf = leaves[test_index]
        root = std_tree.root
        is_valid = MerkleTree.verify_proof(test_leaf, proof, root, hash_fn)
        assert is_valid, "Proof verification failed!"

        print(f"{n:<10} | {format_time(cpu_build_time):<20} | {format_time(batch_build_time):<17} | {format_time(proof_gen_time):<12}")



def run_all_benchmarks():
    # Tree sizes to test
    sizes = [16, 256, 2048, 8192] 
    
    print("Initializing CPU Hash Functions...")
    poseidon_fn = make_poseidon_fn()
    poseidon_batch = make_numpy_batch_fn(poseidon_fn)
    
    rescue_fn = make_rescue_fn()
    rescue_batch = make_numpy_batch_fn(rescue_fn)
    
    print("Initializing GPU CuPy Array Kernel for Rescue-Prime...")
    rescue_gpu_batch = make_pure_cupy_batch_rescue_fn()
    
    # Warm up the GPU (CuPy has a slight overhead on the first execution)
    _ = rescue_gpu_batch(cp.array([1, 2], dtype=cp.uint64), cp.array([3, 4], dtype=cp.uint64))
    
    print("Initialization Complete. Starting Benchmarks...\n")

    # Run CPU vs CPU Benchmarks
    benchmark_hash_function("Poseidon (BN254) - CPU Baseline", poseidon_fn, poseidon_batch, sizes, is_gpu=False)
    benchmark_hash_function("Rescue Prime (32-bit) - CPU Baseline", rescue_fn, rescue_batch, sizes, is_gpu=False)
    
    # Run CPU vs GPU Benchmark
    benchmark_hash_function("Rescue Prime - GPU ACCELERATION", rescue_fn, rescue_gpu_batch, sizes, is_gpu=True)

if __name__ == "__main__":
    random.seed(42)
    run_all_benchmarks()