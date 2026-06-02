import cupy as cp
from hashfunctions.RescuePrime import RescuePrime

def cupy_mod_pow(base_array, exp, mod):
    """
    Computes (base_array ^ exp) % mod for CuPy arrays without overflowing 64-bit integers.
    Uses the square-and-multiply algorithm.
    """
    result = cp.ones_like(base_array, dtype=cp.uint64)
    b = base_array % mod
    e = int(exp)  # Keep exponent as a standard Python int for the loop

    while e > 0:
        if e & 1:  # If exponent is odd
            result = (result * b) % mod
        b = (b * b) % mod
        e >>= 1    # Divide exponent by 2
        
    return result

def make_pure_cupy_batch_rescue_fn(
    p: int = 4_294_967_291,
    m: int = 3,
    capacity: int = 1,
    security_level: int = 80,
):
    """
    Returns a CuPy-vectorised batch hash function for BatchMerkleTree 
    without requiring custom C kernels.
    """
    rp = RescuePrime(p=p, m=m, capacity=capacity, security_level=security_level)
    
    # Transfer constants to GPU once
    p_gpu = cp.uint64(rp.p)
    MDS_gpu = cp.array(rp.MDS, dtype=cp.uint64)
    rc_gpu = cp.array(rp.round_constants, dtype=cp.uint64)
    N_rounds = rp.N

    def cupy_mat_vec_mul(matrix, state_batch):
        """Custom matrix multiplication to prevent uint64 overflow."""
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
            # 1. Forward S-Box (state^3 % p)
            state = (state * state % p_gpu) * state % p_gpu
            
            # 2. MDS Matrix Multiply
            state = cupy_mat_vec_mul(MDS_gpu, state)
            
            # 3. Add Round Constants (First Half)
            rc_step1 = rc_gpu[i * 2 * m : i * 2 * m + m].reshape(m, 1)
            state = (state + rc_step1) % p_gpu
                
            # 4. Inverse S-Box: state^alpha_inv % p (Using our pure CuPy function)
            state = cupy_mod_pow(state, rp.alpha_inv, p_gpu)
                
            # 5. MDS Matrix Multiply
            state = cupy_mat_vec_mul(MDS_gpu, state)
            
            # 6. Add Round Constants (Second Half)
            rc_step2 = rc_gpu[i * 2 * m + m : i * 2 * m + 2 * m].reshape(m, 1)
            state = (state + rc_step2) % p_gpu
                
        return state[1, :]

    return batch_hash

if __name__ == "__main__":
    print("Initializing GPU CuPy Array Kernel for Rescue-Prime...")
    # Use the new pure Python/CuPy function
    rescue_gpu_batch = make_pure_cupy_batch_rescue_fn()