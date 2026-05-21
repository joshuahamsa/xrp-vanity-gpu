/* GPU sieve: pubkey33 -> account_id -> base58check address -> prefix match.
   Self-contained SHA-256 + RIPEMD-160 + base58, NVRTC-compatible (ASCII only,
   no #include, relies on the uint*_t typedefs from vanity/gpu.py's preamble).
   Concatenated after ed25519_kernel.cu / pipeline_kernel.cu so it can reuse the
   Ed25519 device functions for seed -> pubkey derivation. */

/* ------------------------------------------------------------------ SHA-256 */
__device__ __constant__ uint32_t SV_K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

#define SV_ROR32(x,n) (((x) >> (n)) | ((x) << (32 - (n))))

/* len must be <= 55 (single 64-byte block); true for all our inputs (33/32/21).
   __noinline__: keeps sieve_pipeline small so NVRTC's optimizer doesn't blow up
   (inlining all of this into the Ed25519 kernel made compile time explode). */
__noinline__ __device__ void sv_sha256(const uint8_t *msg, uint32_t len, uint8_t out[32]) {
    uint8_t block[64];
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (uint32_t i = 0; i < len; i++) block[i] = msg[i];
    block[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++)
        block[63 - i] = (uint8_t)(bits >> (8 * i));

    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) | (uint32_t)block[i*4+3];
    #pragma unroll 1
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = SV_ROR32(w[i-15],7) ^ SV_ROR32(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = SV_ROR32(w[i-2],17) ^ SV_ROR32(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a;
    uint32_t e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    #pragma unroll 1
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = SV_ROR32(e,6) ^ SV_ROR32(e,11) ^ SV_ROR32(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SV_K256[i] + w[i];
        uint32_t S0 = SV_ROR32(a,2) ^ SV_ROR32(a,13) ^ SV_ROR32(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    uint32_t hv[8];
    hv[0]=0x6a09e667+a; hv[1]=0xbb67ae85+b; hv[2]=0x3c6ef372+c; hv[3]=0xa54ff53a+d;
    hv[4]=0x510e527f+e; hv[5]=0x9b05688c+f; hv[6]=0x1f83d9ab+g; hv[7]=0x5be0cd19+h;
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(hv[i] >> 24);
        out[i*4+1] = (uint8_t)(hv[i] >> 16);
        out[i*4+2] = (uint8_t)(hv[i] >> 8);
        out[i*4+3] = (uint8_t)hv[i];
    }
}

extern "C" __global__
void sha256_test(const uint8_t *in, uint32_t len, uint8_t *out) {
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
        sv_sha256(in, len, out);
}

/* --------------------------------------------------------------- RIPEMD-160 */
#define SV_ROL32(x,n) (((x) << (n)) | ((x) >> (32 - (n))))

__device__ __constant__ int SV_RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
__device__ __constant__ int SV_RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
__device__ __constant__ int SV_SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
__device__ __constant__ int SV_SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
__device__ __constant__ uint32_t SV_KL[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
__device__ __constant__ uint32_t SV_KR[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};

/* len must be <= 55 (single block); always 32 here (a SHA-256 digest). */
__noinline__ __device__ void sv_ripemd160(const uint8_t *msg, uint32_t len, uint8_t out[20]) {
    uint8_t block[64];
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (uint32_t i = 0; i < len; i++) block[i] = msg[i];
    block[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++)
        block[56 + i] = (uint8_t)(bits >> (8 * i));

    uint32_t X[16];
    for (int i = 0; i < 16; i++)
        X[i] = (uint32_t)block[i*4] | ((uint32_t)block[i*4+1] << 8) |
               ((uint32_t)block[i*4+2] << 16) | ((uint32_t)block[i*4+3] << 24);

    uint32_t h0=0x67452301,h1=0xEFCDAB89,h2=0x98BADCFE,h3=0x10325476,h4=0xC3D2E1F0;
    uint32_t al=h0,bl=h1,cl=h2,dl=h3,el=h4;
    uint32_t ar=h0,br=h1,cr=h2,dr=h3,er=h4;
    #pragma unroll 1
    for (int j = 0; j < 80; j++) {
        int rnd = j / 16;
        uint32_t fl, fr, t;
        if (rnd==0) fl = bl ^ cl ^ dl;
        else if (rnd==1) fl = (bl & cl) | (~bl & dl);
        else if (rnd==2) fl = (bl | ~cl) ^ dl;
        else if (rnd==3) fl = (bl & dl) | (cl & ~dl);
        else fl = bl ^ (cl | ~dl);
        t = SV_ROL32(al + fl + X[SV_RL[j]] + SV_KL[rnd], SV_SL[j]) + el;
        al=el; el=dl; dl=SV_ROL32(cl,10); cl=bl; bl=t;
        if (rnd==0) fr = br ^ (cr | ~dr);
        else if (rnd==1) fr = (br & dr) | (cr & ~dr);
        else if (rnd==2) fr = (br | ~cr) ^ dr;
        else if (rnd==3) fr = (br & cr) | (~br & dr);
        else fr = br ^ cr ^ dr;
        t = SV_ROL32(ar + fr + X[SV_RR[j]] + SV_KR[rnd], SV_SR[j]) + er;
        ar=er; er=dr; dr=SV_ROL32(cr,10); cr=br; br=t;
    }
    uint32_t tmp = h1 + cl + dr;
    h1 = h2 + dl + er;
    h2 = h3 + el + ar;
    h3 = h4 + al + br;
    h4 = h0 + bl + cr;
    h0 = tmp;
    uint32_t hv[5] = {h0,h1,h2,h3,h4};
    for (int i = 0; i < 5; i++) {
        out[i*4]   = (uint8_t)hv[i];
        out[i*4+1] = (uint8_t)(hv[i] >> 8);
        out[i*4+2] = (uint8_t)(hv[i] >> 16);
        out[i*4+3] = (uint8_t)(hv[i] >> 24);
    }
}

extern "C" __global__
void ripemd160_test(const uint8_t *in, uint32_t len, uint8_t *out) {
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
        sv_ripemd160(in, len, out);
}

/* ------------------------------------------------------------------- base58 */
__device__ __constant__ char SV_B58[59] =
    "rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz";

/* Encode 25-byte payload (0x00||acct20||cksum4) -> addr (NUL-terminated).
   Returns length. */
__noinline__ __device__ int sv_base58check(const uint8_t payload[25], char *addr) {
    int zeros = 0;
    while (zeros < 25 && payload[zeros] == 0) zeros++;
    uint8_t b58[40];
    int b58len = 0;
    for (int i = zeros; i < 25; i++) {
        int carry = payload[i];
        for (int j = 0; j < b58len; j++) {
            carry += 256 * b58[j];
            b58[j] = carry % 58;
            carry /= 58;
        }
        while (carry) { b58[b58len++] = carry % 58; carry /= 58; }
    }
    int n = 0;
    for (int i = 0; i < zeros; i++) addr[n++] = SV_B58[0];
    for (int i = b58len - 1; i >= 0; i--) addr[n++] = SV_B58[b58[i]];
    addr[n] = '\0';
    return n;
}

/* pubkey33 -> classic address string. Returns length. */
__noinline__ __device__ int sv_address(const uint8_t *pubkey33, char *addr) {
    uint8_t sha[32], acct[20], payload[25], s1[32], s2[32];
    sv_sha256(pubkey33, 33, sha);
    sv_ripemd160(sha, 32, acct);
    payload[0] = 0x00;
    for (int i = 0; i < 20; i++) payload[1 + i] = acct[i];
    sv_sha256(payload, 21, s1);
    sv_sha256(s1, 32, s2);
    for (int i = 0; i < 4; i++) payload[21 + i] = s2[i];
    return sv_base58check(payload, addr);
}

extern "C" __global__
void address_test(const uint8_t *pubkey33, uint8_t *out_addr, uint8_t *out_len) {
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        char addr[48];
        int n = sv_address(pubkey33, addr);
        for (int i = 0; i < n; i++) out_addr[i] = (uint8_t)addr[i];
        *out_len = (uint8_t)n;
    }
}

/* --------------------------------------------------------- pubkey sieve */
/* pubkey33 -> address -> prefix compare; emit matching indices. Kept separate
   from the Ed25519 pipeline kernel so this stays small and compiles fast.
   needle is pre-lowercased by the host when case_sensitive == 0. */
extern "C" __global__
void sieve_pubkeys(
    const unsigned char * __restrict__ pubkeys,  /* B * 33 */
    const unsigned char * __restrict__ needle,   /* needle_len */
    unsigned int needle_len,
    unsigned int case_sensitive,
    unsigned int * __restrict__ out_indices,     /* capacity max_out */
    unsigned int * __restrict__ out_count,       /* single counter */
    unsigned int max_out,
    unsigned int B
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    char addr[48];
    int alen = sv_address(pubkeys + i * 33, addr);
    if ((unsigned int)alen < 1 + needle_len) return;

    for (unsigned int k = 0; k < needle_len; k++) {
        char ch = addr[1 + k];
        if (!case_sensitive && ch >= 'A' && ch <= 'Z') ch += 32;
        if (ch != (char)needle[k]) return;
    }
    unsigned int slot = atomicAdd(out_count, 1);
    if (slot < max_out) out_indices[slot] = i;
}
