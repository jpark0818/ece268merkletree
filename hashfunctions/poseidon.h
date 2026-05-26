/*
 * poseidon.h — Public API for the Poseidon hash function (BN254 scalar field)
 *
 * Include this header in any C program that uses poseidon.c.
 * Link with:  gcc your_program.c poseidon.c -lgmp -o your_program
 */

#ifndef POSEIDON_H
#define POSEIDON_H

#include <gmp.h>

/* ============================================================
 * Parameters (read-only, do not modify)
 * ============================================================ */

#define POSEIDON_T         3    /* state width: 1 capacity + 2 rate elements  */
#define POSEIDON_R_F       8    /* full rounds                                 */
#define POSEIDON_R_P       57   /* partial rounds                              */
#define POSEIDON_N_ROUNDS  65   /* total rounds  (R_F + R_P)                  */
#define POSEIDON_ALPHA     5    /* S-box exponent  x^5 mod p                  */
#define POSEIDON_N_BITS    254  /* BN254 prime field bit-width                 */

/* BN254 scalar field prime (decimal string, 254-bit) */
#define POSEIDON_PRIME_STR \
    "21888242871839275222246405745257275088548364400416034343698204186575808495617"

/* ============================================================
 * Lifecycle
 * ============================================================
 *
 * poseidon_init()    must be called once before any hash call.
 *                    Generates all round constants and the MDS matrix.
 *                    Safe to call multiple times (no-op after first call).
 *
 * poseidon_cleanup() releases all memory allocated by poseidon_init().
 *                    Call when the library is no longer needed.
 */
void poseidon_init(void);
void poseidon_cleanup(void);

/* ============================================================
 * Core hash  —  two field elements → one field element
 * ============================================================
 *
 *   poseidon_hash(in0, in1, out)
 *
 *   Computes the Poseidon 2-to-1 hash using the sponge construction:
 *     state  = [ 0,   in0,  in1 ]   (capacity=0, rate=[in0,in1])
 *     state' = poseidon_permute(state)
 *     out    = state'[1]
 *
 *   Parameters
 *     in0, in1  [in]  Field elements.  Must satisfy  0 <= x < p.
 *     out       [out] Initialised mpz_t that receives the result.
 *                     Also satisfies  0 <= out < p.
 *
 *   Typical usage (Merkle tree parent node):
 *     mpz_t left, right, parent;
 *     mpz_inits(left, right, parent, NULL);
 *     mpz_set_str(left,  "...", 16);
 *     mpz_set_str(right, "...", 16);
 *     poseidon_hash(left, right, parent);
 *     mpz_clears(left, right, parent, NULL);
 */
void poseidon_hash(const mpz_t in0, const mpz_t in1, mpz_t out);

/* ============================================================
 * Hex-string convenience wrapper
 * ============================================================
 *
 *   poseidon_hash_hex(hex0, hex1, hex_out)
 *
 *   Same as poseidon_hash() but accepts and produces hex strings.
 *   Useful when field elements are stored as hex (e.g. from files
 *   or other libraries).
 *
 *   Parameters
 *     hex0      [in]  Null-terminated hex string for the first input.
 *                     May include leading zeros.  Max 64 hex digits.
 *     hex1      [in]  Null-terminated hex string for the second input.
 *     hex_out   [out] Caller-allocated buffer of at least 65 bytes.
 *                     Receives a zero-padded 64-character hex string
 *                     (no "0x" prefix) followed by a null terminator.
 *
 *   Example
 *     char result[65];
 *     poseidon_hash_hex(
 *         "0000000000000000000000000000000000000000000000000000000000000001",
 *         "0000000000000000000000000000000000000000000000000000000000000002",
 *         result);
 *     printf("%s\n", result);   // 64 hex digits
 */
void poseidon_hash_hex(const char *hex0, const char *hex1, char *hex_out);

/* ============================================================
 * Raw permutation  (advanced use)
 * ============================================================
 *
 *   poseidon_permute(state)
 *
 *   Applies the Poseidon-π permutation directly to a T-element state.
 *   Use this when building custom sponge modes (e.g. hashing more
 *   than 2 inputs, or absorbing a domain-separation tag).
 *
 *   Parameters
 *     state  [in/out]  Array of T initialised mpz_t values.
 *                      Each element must satisfy  0 <= x < p.
 *                      Modified in place.
 *
 *   Example (manual sponge for 2-input hash):
 *     mpz_t state[POSEIDON_T];
 *     mpz_init_set_ui(state[0], 0);    // capacity
 *     mpz_init_set   (state[1], in0);  // rate[0]
 *     mpz_init_set   (state[2], in1);  // rate[1]
 *     poseidon_permute(state);
 *     // state[1] is now the hash output
 *     for (int i = 0; i < POSEIDON_T; i++) mpz_clear(state[i]);
 */
void poseidon_permute(mpz_t state[POSEIDON_T]);

#endif /* POSEIDON_H */
