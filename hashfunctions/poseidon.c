/*
 * poseidon.c — Poseidon hash function over the BN254 scalar field
 *
 * Parameters : t = 3  (2-to-1 hash)
 *              R_F = 8  full rounds  (4 at start, 4 at end)
 *              R_P = 57 partial rounds
 *              alpha = 5  (S-box: x^5 mod p)
 *
 * Reference  : "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems"
 *              Grassi, Khovratovich, Rechberger, Roy, Schofnegger — 2019
 *              https://eprint.iacr.org/2019/458
 *
 * Build      : gcc -O2 -Wall -o poseidon poseidon.c -lgmp
 * Dependency : sudo apt-get install libgmp-dev
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <gmp.h>

/* ============================================================
 * Parameters
 * ============================================================ */

#define T         3               /* state width (capacity 1 + rate 2) */
#define R_F       8               /* full rounds                        */
#define R_P       57              /* partial rounds                     */
#define N_ROUNDS  (R_F + R_P)    /* 65 total                           */
#define ALPHA     5               /* S-box exponent                     */
#define N_BITS    254             /* BN254 prime field bit-width        */

/* BN254 scalar field prime r (decimal) */
static const char PRIME_STR[] =
    "21888242871839275222246405745257275088548364400416034343698204186575808495617";

/* ============================================================
 * Grain LFSR — deterministic round-constant generation
 *
 * 80-bit LFSR with feedback polynomial x^80+x^62+x^51+x^38+x^23+x^13+1.
 * The initialization vector encodes all Poseidon parameters so that
 * different parameter sets yield independent constants (Section 4.2).
 * ============================================================ */

typedef struct { uint8_t s[80]; } Grain;

/* One LFSR clock.  During the 160-step warm-up the output bit is XOR'd
 * back into the feedback so the initial state is fully absorbed. */
static uint8_t grain_clock(Grain *g, int warmup)
{
    /* s[0] = oldest bit; feedback taps at positions 0,13,23,38,51,62 */
    uint8_t out = g->s[0];
    uint8_t fb  = g->s[0] ^ g->s[13] ^ g->s[23]
                ^ g->s[38] ^ g->s[51] ^ g->s[62];
    if (warmup) fb ^= out;          /* absorb output during init       */
    memmove(g->s, g->s + 1, 79);
    g->s[79] = fb;
    return out;
}

/* Two-step rejection filter that removes bias from the raw LFSR stream. */
static uint8_t grain_bit(Grain *g)
{
    uint8_t b0, b1;
    do {
        b0 = grain_clock(g, 0);
        b1 = grain_clock(g, 0);
    } while (!b0);
    return b1;
}

/* Rejection-sample one field element uniformly from [0, prime). */
static void grain_field(Grain *g, const mpz_t prime, mpz_t out)
{
    for (;;) {
        mpz_set_ui(out, 0);
        /* MSB first: first bit from grain_bit() lands in position 2^253 */
        for (int i = 0; i < N_BITS; i++) {
            mpz_mul_2exp(out, out, 1);
            if (grain_bit(g)) mpz_add_ui(out, out, 1);
        }
        if (mpz_cmp(out, prime) < 0) return;   /* accept */
    }
}

/* Build and warm-up the Grain LFSR from the Poseidon parameter set. */
static void grain_init(Grain *g)
{
    memset(g->s, 0, sizeof g->s);

    /*  Bit layout (80 bits total):
     *   [0- 1]  = 1,0           field type: prime
     *   [2- 5]  = ALPHA         4 bits, MSB first
     *   [6-17]  = N_BITS        12 bits, MSB first
     *  [18-29]  = T             12 bits, MSB first
     *  [30-39]  = R_F           10 bits, MSB first
     *  [40-49]  = R_P           10 bits, MSB first
     *  [50-79]  = 1…1           30 padding ones
     */
    g->s[0] = 1; /* s[1] stays 0 */

    for (int i = 0; i < 4;  i++) g->s[2  + i] = (ALPHA  >> (3  - i)) & 1;
    for (int i = 0; i < 12; i++) g->s[6  + i] = (N_BITS >> (11 - i)) & 1;
    for (int i = 0; i < 12; i++) g->s[18 + i] = (T      >> (11 - i)) & 1;
    for (int i = 0; i < 10; i++) g->s[30 + i] = (R_F    >> (9  - i)) & 1;
    for (int i = 0; i < 10; i++) g->s[40 + i] = (R_P    >> (9  - i)) & 1;
    for (int i = 50; i < 80; i++) g->s[i] = 1;

    for (int i = 0; i < 160; i++) grain_clock(g, 1);   /* warm-up */
}

/* ============================================================
 * Global Poseidon state
 * ============================================================ */

static mpz_t p;                    /* field prime                      */
static mpz_t RC[N_ROUNDS][T];     /* round constants [round][position] */
static mpz_t MDS[T][T];           /* MDS matrix                       */
static int   ready = 0;

/* ============================================================
 * MDS matrix — Cauchy construction (guaranteed MDS)
 *
 * M[i][j] = 1 / (x_i - y_j)  mod p
 * x = {0, 1, …, T-1}   y = {T, T+1, …, 2T-1}
 *
 * Because every x_i < T <= y_j, we have x_i ≠ y_j for all (i,j).
 * With all x values distinct and all y values distinct, every square
 * sub-matrix of a Cauchy matrix is invertible, so M is MDS.
 * ============================================================ */

static void build_mds(void)
{
    mpz_t xi, yj, d;
    mpz_inits(xi, yj, d, NULL);

    for (int i = 0; i < T; i++) {
        mpz_set_ui(xi, (unsigned)i);
        for (int j = 0; j < T; j++) {
            mpz_set_ui(yj, (unsigned)(T + j));
            mpz_sub(d, xi, yj);         /* negative, but mod will fix it */
            mpz_mod(d, d, p);
            mpz_invert(MDS[i][j], d, p);
        }
    }

    mpz_clears(xi, yj, d, NULL);
}

/* ============================================================
 * Initialization / cleanup
 * ============================================================ */

void poseidon_init(void)
{
    if (ready) return;

    mpz_init_set_str(p, PRIME_STR, 10);

    /* Generate N_ROUNDS * T round constants with the Grain LFSR */
    Grain g;
    grain_init(&g);
    for (int r = 0; r < N_ROUNDS; r++)
        for (int i = 0; i < T; i++) {
            mpz_init(RC[r][i]);
            grain_field(&g, p, RC[r][i]);
        }

    /* Build MDS matrix */
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
            mpz_init(MDS[i][j]);
    build_mds();

    ready = 1;
}

void poseidon_cleanup(void)
{
    if (!ready) return;
    for (int r = 0; r < N_ROUNDS; r++)
        for (int i = 0; i < T; i++)
            mpz_clear(RC[r][i]);
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
            mpz_clear(MDS[i][j]);
    mpz_clear(p);
    ready = 0;
}

/* ============================================================
 * Poseidon permutation internals
 * ============================================================ */

static void add_round_constants(mpz_t state[T], int round)
{
    for (int i = 0; i < T; i++) {
        mpz_add(state[i], state[i], RC[round][i]);
        mpz_mod(state[i], state[i], p);
    }
}

static void sbox_full(mpz_t state[T])
{
    for (int i = 0; i < T; i++)
        mpz_powm_ui(state[i], state[i], ALPHA, p);
}

static void sbox_partial(mpz_t state[T])
{
    mpz_powm_ui(state[0], state[0], ALPHA, p);   /* only state[0] */
}

static void mds_multiply(mpz_t state[T])
{
    mpz_t acc[T], tmp;
    mpz_init(tmp);
    for (int i = 0; i < T; i++) mpz_init_set_ui(acc[i], 0);

    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++) {
            mpz_mul(tmp, MDS[i][j], state[j]);
            mpz_add(acc[i], acc[i], tmp);
        }

    for (int i = 0; i < T; i++) {
        mpz_mod(state[i], acc[i], p);
        mpz_clear(acc[i]);
    }
    mpz_clear(tmp);
}

/* ============================================================
 * Poseidon-π permutation
 *
 * Structure:
 *   R_F/2 full rounds  →  R_P partial rounds  →  R_F/2 full rounds
 * Each round: AddRoundConstants → S-box → MDS
 * ============================================================ */

void poseidon_permute(mpz_t state[T])
{
    assert(ready);
    const int half = R_F / 2;   /* = 4 */

    /* First R_F/2 full rounds */
    for (int r = 0; r < half; r++) {
        add_round_constants(state, r);
        sbox_full(state);
        mds_multiply(state);
    }

    /* R_P partial rounds */
    for (int r = half; r < half + R_P; r++) {
        add_round_constants(state, r);
        sbox_partial(state);
        mds_multiply(state);
    }

    /* Last R_F/2 full rounds */
    for (int r = half + R_P; r < N_ROUNDS; r++) {
        add_round_constants(state, r);
        sbox_full(state);
        mds_multiply(state);
    }
}

/* ============================================================
 * Poseidon hash  (2-to-1 sponge)
 *
 * Capacity element : state[0] = 0
 * Rate elements    : state[1] = in0,  state[2] = in1
 * Output           : state[1] after the permutation
 *
 * Both inputs must be valid field elements (0 <= x < p).
 * ============================================================ */

void poseidon_hash(const mpz_t in0, const mpz_t in1, mpz_t out)
{
    assert(ready);
    mpz_t state[T];
    mpz_init_set_ui(state[0], 0);   /* capacity */
    mpz_init_set(state[1], in0);    /* rate[0]  */
    mpz_init_set(state[2], in1);    /* rate[1]  */

    poseidon_permute(state);

    mpz_set(out, state[1]);

    for (int i = 0; i < T; i++) mpz_clear(state[i]);
}

/* Convenience wrapper: accept / produce 64-character hex strings.
 * hex_out must point to a buffer of at least 65 bytes. */
void poseidon_hash_hex(const char *hex0, const char *hex1, char *hex_out)
{
    mpz_t a, b, h;
    mpz_init_set_str(a, hex0, 16);
    mpz_init_set_str(b, hex1, 16);
    mpz_init(h);

    poseidon_hash(a, b, h);

    /* Zero-pad to 64 hex digits */
    char buf[65];
    gmp_sprintf(buf, "%064Zx", h);
    memcpy(hex_out, buf, 65);

    mpz_clears(a, b, h, NULL);
}

/* ============================================================
 * Test helpers
 * ============================================================ */

/* Count the number of bit positions that differ between a and b. */
static mp_bitcnt_t count_differing_bits(const mpz_t a, const mpz_t b)
{
    mpz_t xv;
    mpz_init(xv);
    mpz_xor(xv, a, b);
    mp_bitcnt_t diff = mpz_popcount(xv);
    mpz_clear(xv);
    return diff;
}

static void report(const char *desc, int pass)
{
    printf("    %-52s [%s]\n", desc, pass ? "PASS" : "FAIL");
}

/* ============================================================
 * Test 1 — Known-answer / regression
 *
 * For each fixed input pair the test checks:
 *   (a) output is a valid BN254 field element: 0 <= h < p
 *   (b) output is non-zero  (trivially fails only with prob 1/p)
 *   (c) output is deterministic: two calls produce the same value
 *
 * The printed 64-hex-digit values serve as golden references.
 * Record them after the first run and hard-code them here to
 * catch any future implementation regression.
 *
 * Cross-verification with circomlib / iden3 is not possible
 * directly because those libraries use a different MDS matrix;
 * however the structural properties above must hold for any
 * correct Poseidon instantiation over BN254.
 * ============================================================ */
static int test_known_answer(void)
{
    printf("\n[Test 1] Known-answer / regression\n");

    static const struct { unsigned long a; unsigned long b; } cases[] = {
        {0,   0  },
        {1,   2  },
        {3,   4  },
        {100, 200},
    };
    const int n = (int)(sizeof cases / sizeof cases[0]);

    mpz_t a, b, h1, h2;
    mpz_inits(a, b, h1, h2, NULL);
    int all_pass = 1;

    for (int k = 0; k < n; k++) {
        mpz_set_ui(a, cases[k].a);
        mpz_set_ui(b, cases[k].b);

        poseidon_hash(a, b, h1);
        poseidon_hash(a, b, h2);        /* second call must match */

        int valid   = (mpz_sgn(h1) >= 0) && (mpz_cmp(h1, p) < 0);
        int nonzero = (mpz_sgn(h1) != 0);
        int determ  = (mpz_cmp(h1, h2) == 0);

        gmp_printf("  poseidon(%3lu, %3lu) = %064Zx\n",
                   cases[k].a, cases[k].b, h1);

        char buf[64];
        snprintf(buf, sizeof buf, "output in [0, p)  for (%lu,%lu)",
                 cases[k].a, cases[k].b);
        report(buf, valid);
        snprintf(buf, sizeof buf, "output != 0       for (%lu,%lu)",
                 cases[k].a, cases[k].b);
        report(buf, nonzero);
        snprintf(buf, sizeof buf, "deterministic     for (%lu,%lu)",
                 cases[k].a, cases[k].b);
        report(buf, determ);

        if (!valid || !nonzero || !determ) all_pass = 0;
    }

    mpz_clears(a, b, h1, h2, NULL);
    printf("  => Test 1 overall: %s\n", all_pass ? "PASSED" : "FAILED");
    return all_pass;
}

/* ============================================================
 * Test 2 — Avalanche effect
 *
 * A one-bit change in the input should flip roughly half of the
 * output bits.  We test: for each pair (a, b), compare
 *   h1 = poseidon(a,   b)
 *   h2 = poseidon(a+1, b)
 * and count differing bits via XOR + popcount.
 *
 * Threshold: we require at least N_BITS/4 = 63 bits to differ,
 * which is a conservative lower bound; a good hash should
 * average ~127 (50 %).
 * ============================================================ */
static int test_avalanche(void)
{
    printf("\n[Test 2] Avalanche effect  (1-bit input change -> ~50%% output change)\n");

    static const struct { unsigned long a; unsigned long b; } cases[] = {
        {1,   2  },
        {42,  99 },
        {0,   0  },
        {999, 1000},
    };
    const int n = (int)(sizeof cases / sizeof cases[0]);

    mpz_t a, b, h1, h2;
    mpz_inits(a, b, h1, h2, NULL);
    int all_pass = 1;

    const mp_bitcnt_t threshold = N_BITS / 4;   /* >= 63 bits must differ */

    for (int k = 0; k < n; k++) {
        mpz_set_ui(a, cases[k].a);
        mpz_set_ui(b, cases[k].b);
        poseidon_hash(a, b, h1);

        /* flip LSB of a: a' = (a + 1) mod p */
        mpz_add_ui(a, a, 1);
        mpz_mod(a, a, p);
        poseidon_hash(a, b, h2);

        mp_bitcnt_t diff = count_differing_bits(h1, h2);
        double pct = 100.0 * (double)diff / N_BITS;
        int pass = (diff >= threshold);

        printf("  case %d  a=%lu, b=%lu:  %lu / %d bits differ  (%.1f%%)  [%s]\n",
               k + 1, cases[k].a, cases[k].b,
               (unsigned long)diff, N_BITS, pct,
               pass ? "PASS" : "FAIL");

        if (!pass) all_pass = 0;
    }

    mpz_clears(a, b, h1, h2, NULL);
    printf("  => Test 2 overall: %s\n", all_pass ? "PASSED" : "FAILED");
    return all_pass;
}

/* ============================================================
 * Test 3 — Non-commutativity  hash(a,b) != hash(b,a)
 *
 * In a Merkle tree the left and right child are ordered: swapping
 * them must produce a different parent hash, otherwise an attacker
 * could present a mirror-image tree as a valid proof.
 *
 * The MDS matrix is not symmetric and the sponge inputs are
 * loaded in order, so poseidon(a,b) != poseidon(b,a) whenever a!=b.
 * ============================================================ */
static int test_noncommutative(void)
{
    printf("\n[Test 3] Non-commutativity  poseidon(a,b) != poseidon(b,a)\n");

    static const struct { unsigned long a; unsigned long b; } cases[] = {
        {1,   2   },
        {3,   7   },
        {100, 200 },
        {999, 1000},
    };
    const int n = (int)(sizeof cases / sizeof cases[0]);

    mpz_t a, b, hab, hba;
    mpz_inits(a, b, hab, hba, NULL);
    int all_pass = 1;

    for (int k = 0; k < n; k++) {
        mpz_set_ui(a, cases[k].a);
        mpz_set_ui(b, cases[k].b);

        poseidon_hash(a, b, hab);
        poseidon_hash(b, a, hba);

        int diff = (mpz_cmp(hab, hba) != 0);

        char buf[64];
        snprintf(buf, sizeof buf,
                 "poseidon(%lu,%lu) != poseidon(%lu,%lu)",
                 cases[k].a, cases[k].b,
                 cases[k].b, cases[k].a);
        report(buf, diff);
        if (!diff) all_pass = 0;
    }

    mpz_clears(a, b, hab, hba, NULL);
    printf("  => Test 3 overall: %s\n", all_pass ? "PASSED" : "FAILED");
    return all_pass;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void)
{
    printf("============================================================\n");
    printf(" Poseidon hash  BN254  t=%d  R_F=%d  R_P=%d  alpha=%d\n",
           T, R_F, R_P, ALPHA);
    printf("============================================================\n");
    printf("Generating %d round constants via Grain LFSR …\n", N_ROUNDS * T);
    poseidon_init();
    printf("Done.\n");

    int r1 = test_known_answer();
    int r2 = test_avalanche();
    int r3 = test_noncommutative();

    printf("\n============================================================\n");
    printf(" Summary:  Test1=%s  Test2=%s  Test3=%s\n",
           r1 ? "PASS" : "FAIL",
           r2 ? "PASS" : "FAIL",
           r3 ? "PASS" : "FAIL");
    printf("============================================================\n");

    poseidon_cleanup();
    return (r1 && r2 && r3) ? 0 : 1;
}
