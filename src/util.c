#include "util.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

/* ------------------------------------------------------------------ logs */

static void log_to(FILE *stream, const char *prefix, const char *fmt, va_list ap) {
    fputs(prefix, stream);
    vfprintf(stream, fmt, ap);
    fputc('\n', stream);
    fflush(stream);
}

void aster_info(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); log_to(stdout, "  ", fmt, ap); va_end(ap);
}

void aster_warn(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); log_to(stderr, "warning: ", fmt, ap); va_end(ap);
}

void aster_fail(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); log_to(stderr, "error: ", fmt, ap); va_end(ap);
    exit(1);
}

/* ------------------------------------------------------------ allocation */

size_t aster_size_mul(size_t a, size_t b) {
    if (a != 0 && b > (size_t)-1 / a) aster_fail("size overflow computing %zu * %zu", a, b);
    return a * b;
}

size_t aster_size_add(size_t a, size_t b) {
    if (b > (size_t)-1 - a) aster_fail("size overflow computing %zu + %zu", a, b);
    return a + b;
}

void *aster_xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) aster_fail("out of memory allocating %zu bytes", bytes);
    return p;
}

void *aster_xcalloc(size_t count, size_t size) {
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) aster_fail("out of memory allocating %zu x %zu bytes", count, size);
    return p;
}

void *aster_xrealloc(void *ptr, size_t bytes) {
    void *p = realloc(ptr, bytes ? bytes : 1);
    if (!p) aster_fail("out of memory reallocating %zu bytes", bytes);
    return p;
}

/* ----------------------------------------------------------------- files */

char *aster_read_file(const char *path, size_t *out_len, int *out_ok) {
    if (out_len) *out_len = 0;
    if (out_ok) *out_ok = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long size = ftell(f);
    if (size < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)aster_xmalloc((size_t)size + 1);
    size_t got = size ? fread(buf, 1, (size_t)size, f) : 0;
    int bad = ferror(f);
    fclose(f);
    if (bad) { free(buf); return NULL; }
    buf[got] = '\0';
    if (out_len) *out_len = got;
    if (out_ok) *out_ok = 1;
    return buf;
}

int aster_file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

void aster_mkdir_for_file(const char *path) {
    char tmp[1024];
    size_t n = strlen(path);
    if (n == 0 || n >= sizeof tmp) return;
    memcpy(tmp, path, n + 1);

    /* Strip the final component, then create each parent level. */
    for (size_t i = n; i-- > 0;) {
        if (tmp[i] == '/' || tmp[i] == '\\') {
            tmp[i] = '\0';
            for (size_t j = 1; j < strlen(tmp); ++j) {
                if (tmp[j] == '/' || tmp[j] == '\\') {
                    tmp[j] = '\0';
#ifdef _WIN32
                    _mkdir(tmp);
#else
                    mkdir(tmp, 0755);
#endif
                    tmp[j] = '/';
                }
            }
#ifdef _WIN32
            _mkdir(tmp);
#else
            mkdir(tmp, 0755);
#endif
            return;
        }
    }
}

void aster_write_file_atomic(const char *path, const void *data, size_t len) {
    aster_mkdir_for_file(path);
    char tmp[1088];
    int n = snprintf(tmp, sizeof tmp, "%s.tmp", path);
    if (n < 0 || (size_t)n >= sizeof tmp) aster_fail("path too long: %s", path);

    FILE *f = fopen(tmp, "wb");
    if (!f) aster_fail("cannot open %s for writing", tmp);
    if (len && fwrite(data, 1, len, f) != len) {
        fclose(f);
        remove(tmp);
        aster_fail("short write to %s", tmp);
    }
    if (fclose(f) != 0) { remove(tmp); aster_fail("cannot flush %s", tmp); }

#ifdef _WIN32
    /* rename() on Windows refuses an existing target; MoveFileEx replaces. */
    if (!MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        remove(tmp);
        aster_fail("cannot replace %s (windows error %lu)", path, (unsigned long)GetLastError());
    }
#else
    if (rename(tmp, path) != 0) { remove(tmp); aster_fail("cannot rename %s into place", tmp); }
#endif
}

/* ---------------------------------------------------------------- sha256 */

typedef struct { uint32_t h[8]; uint64_t bits; unsigned char buf[64]; size_t n; } sha256_t;

static const uint32_t SHA_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

static uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_block(sha256_t *s, const unsigned char *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) |
               ((uint32_t)p[i*4+2] << 8) | (uint32_t)p[i*4+3];
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3];
    uint32_t e = s->h[4], f = s->h[5], g = s->h[6], h = s->h[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + SHA_K[i] + w[i];
        uint32_t S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}

void aster_sha256_hex(const void *data, size_t len, char out[65]) {
    sha256_t s;
    s.h[0]=0x6a09e667u; s.h[1]=0xbb67ae85u; s.h[2]=0x3c6ef372u; s.h[3]=0xa54ff53au;
    s.h[4]=0x510e527fu; s.h[5]=0x9b05688cu; s.h[6]=0x1f83d9abu; s.h[7]=0x5be0cd19u;
    s.bits = (uint64_t)len * 8u; s.n = 0;

    const unsigned char *p = (const unsigned char *)data;
    size_t i = 0;
    for (; i + 64 <= len; i += 64) sha256_block(&s, p + i);

    size_t rest = len - i;
    memcpy(s.buf, p + i, rest);
    s.buf[rest++] = 0x80;
    if (rest > 56) {
        memset(s.buf + rest, 0, 64 - rest);
        sha256_block(&s, s.buf);
        rest = 0;
        memset(s.buf, 0, 56);
    } else {
        memset(s.buf + rest, 0, 56 - rest);
    }
    for (int b = 0; b < 8; ++b) s.buf[56 + b] = (unsigned char)(s.bits >> (56 - 8 * b));
    sha256_block(&s, s.buf);

    for (int b = 0; b < 8; ++b) sprintf(out + b * 8, "%08x", s.h[b]);
    out[64] = '\0';
}

/* ----------------------------------------------------------------- crc32 */

uint32_t aster_crc32(const void *data, size_t len) {
    static uint32_t table[256];
    static int ready = 0;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        ready = 1;
    }
    const unsigned char *p = (const unsigned char *)data;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* ------------------------------------------------------------------ time */

double aster_now_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq; static int ready = 0;
    if (!ready) { QueryPerformanceFrequency(&freq); ready = 1; }
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* -------------------------------------------------------------------- rng */
/* xorshift32: small, deterministic, dependency-free. It is not cryptographic
 * and is only used for weight init, shuffling, and sampling. */

void aster_rng_seed_init(uint32_t *state, uint32_t seed) {
    *state = seed ? seed : 0x9E3779B9u;
    for (int i = 0; i < 8; ++i) (void)aster_rng_u32(state);
}

uint32_t aster_rng_u32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

float aster_rng_uniform(uint32_t *state) {
    return (float)(aster_rng_u32(state) >> 8) / 16777216.0f;  /* [0,1) */
}

float aster_rng_normal(uint32_t *state) {
    /* Box-Muller. u1 is nudged off zero so log() stays finite. */
    float u1 = aster_rng_uniform(state);
    float u2 = aster_rng_uniform(state);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}
