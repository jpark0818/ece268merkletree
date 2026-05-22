import hashlib
import math

class RescuePrime:
    def __init__(self, p: int, m: int, capacity: int, security_level: int):
        """
        Initializes the Rescue-Prime hash function parameters.
        
        :param p: The prime field modulus (must be >= 32 bits)
        :param m: The total state width (number of field elements)
        :param capacity: The capacity of the sponge construction
        :param security_level: Target security level in bits
        """
        self.p = p
        self.m = m
        self.capacity = capacity
        self.rate = m - capacity
        self.security_level = security_level
        
        self.alpha, self.alpha_inv = self._get_alphas()
        
        self.N = self._get_number_of_rounds()
        
        self.MDS = self._get_mds_matrix()
        self.round_constants = self._get_round_constants()

    def _get_alphas(self):
        """Finds the smallest alpha coprime to p-1, and its inverse."""
        alpha = 3
        while math.gcd(alpha, self.p - 1) != 1:
            alpha += 1
        alpha_inv = pow(alpha, -1, self.p - 1)
        return alpha, alpha_inv

    def _get_number_of_rounds(self) -> int:
        """Calculates the necessary rounds to resist Gröbner basis attacks."""
        target = 2 ** self.security_level
        
        for l1 in range(1, 25):
            dcon = int(0.5 * (self.alpha - 1) * self.m * (l1 - 1)) + 2
            v = self.m * (l1 - 1) + self.rate
            
            if math.comb(v + dcon, v) ** 2 > target:
                break
        
        return math.ceil(1.5 * max(5, l1))

    def _get_mds_matrix(self):
        """Generates an m x m MDS matrix using a Vandermonde-matrix approach."""
        g = 2
        while True:
            if pow(g, self.p - 1, self.p) == 1:
                if pow(g, (self.p - 1) // 2, self.p) != 1:
                    break
            g += 1

        V = [[pow(g, i * j, self.p) for j in range(2 * self.m)] for i in range(self.m)]
        
        for i in range(self.m):
            inv = pow(V[i][i], -1, self.p)
            for j in range(2 * self.m):
                V[i][j] = (V[i][j] * inv) % self.p
            
            for k in range(self.m):
                if i != k:
                    factor = V[k][i]
                    for j in range(2 * self.m):
                        V[k][j] = (V[k][j] - factor * V[i][j]) % self.p
                        
        M_T = [row[self.m:] for row in V]
        MDS = [[M_T[j][i] for j in range(self.m)] for i in range(self.m)]
        return MDS

    def _get_round_constants(self):
        """Generates 2 * m * N pseudo-random field constants using SHAKE-256."""
        bytes_per_int = math.ceil(len(bin(self.p)[2:]) / 8) + 1
        num_bytes = bytes_per_int * (2 * self.m * self.N)
        
        seed_string = f"Rescue-XLIX({self.p},{self.m},{self.capacity},{self.security_level})"
        shake = hashlib.shake_256(seed_string.encode('ascii'))
        byte_string = shake.digest(num_bytes)
        
        constants = []
        for i in range(2 * self.m * self.N):
            chunk = byte_string[bytes_per_int * i : bytes_per_int * (i + 1)]
            integer_val = int.from_bytes(chunk, byteorder='little')
            constants.append(integer_val % self.p)
            
        return constants

    def _mat_vec_mul(self, matrix, vector):
        """Helper for matrix-vector multiplication over F_p."""
        result = [0] * self.m
        for i in range(self.m):
            s = 0
            for j in range(self.m):
                s += matrix[i][j] * vector[j]
            result[i] = s % self.p
        return result

    def rescue_xlix_permutation(self, state):
        """Executes N rounds of the core Rescue-XLIX permutation on the state."""
        for i in range(self.N):
            for j in range(self.m):
                state[j] = pow(state[j], self.alpha, self.p)
            
            state = self._mat_vec_mul(self.MDS, state)
            
            for j in range(self.m):
                state[j] = (state[j] + self.round_constants[i * 2 * self.m + j]) % self.p
                
            for j in range(self.m):
                state[j] = pow(state[j], self.alpha_inv, self.p)
                
            state = self._mat_vec_mul(self.MDS, state)
            
            for j in range(self.m):
                state[j] = (state[j] + self.round_constants[i * 2 * self.m + self.m + j]) % self.p
                
        return state

    def hash(self, input_sequence: list[int]) -> list[int]:
        """
        Pads the input stream, processes it via an arithmetic sponge construction,
        and returns the resulting squeezed field elements.
        """
        padded_input = list(input_sequence)
        
        padded_input.append(1)
        while len(padded_input) % self.rate != 0:
            padded_input.append(0)
            
        state = [0] * self.m
        
        absorb_index = 0
        while absorb_index < len(padded_input):
            for i in range(self.rate):
                state[i] = (state[i] + padded_input[absorb_index]) % self.p
                absorb_index += 1
            state = self.rescue_xlix_permutation(state)
            
        output_sequence = []
        for i in range(self.rate):
            output_sequence.append(state[i])
            
        return output_sequence

if __name__ == "__main__":
    PRIME = 4294967291  # 2^32 - 5
    WIDTH = 3
    CAPACITY = 1
    SECURITY = 80
    
    rp_hasher = RescuePrime(p=PRIME, m=WIDTH, capacity=CAPACITY, security_level=SECURITY)
    
    print(f"Instantiated Rescue-Prime successfully!")
    print(f"Calculated S-Box Alpha: {rp_hasher.alpha}")
    print(f"Calculated Total Rounds (N): {rp_hasher.N}")
    print(f"Rate (elements absorbed per round): {rp_hasher.rate}")
    
    my_data = [12345, 789012, 345678, 901234]
    
    # Calculate hash output
    hash_output = rp_hasher.hash(my_data)
    print(f"\nInput Data: {my_data}")
    print(f"Rescue-Prime Hash Output Elements: {hash_output}")