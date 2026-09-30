/* util.h - allocation, logging, file IO, hashing, timing, seeded RNG.
 *
 * Everything here is dependency-free C11 so the project builds with a bare
 * MinGW-w64 GCC on Windows and with clang/gcc on POSIX.
 */
#ifndef ASTER_UTIL_H
#define ASTER_UTIL_H

#include <stddef.h>
#include <stdint.h>

/* ---- logging ---------------------------------------------------------- */
/* Normal logs go to stdout, problems to stderr. Neither prints prompts or
 * dataset contents; see README "Privacy". */
void aster_info(const char *fmt, ...);
void aster_warn(const char *fmt, ...);
void aster_fail(const char *fmt, ...); /* prints and exits with code 1 */

/* ---- checked allocation ----------------------------------------------- */
/* These report a clear message and exit(2) rather than returning NULL, so
 * call sites stay readable. Sizes are checked for overflow first. */
size_t aster_size_mul(size_t a, size_t b);        /* aborts on overflow */
size_t aster_size_add(size_t a, size_t b);
void  *aster_xmalloc(size_t bytes);
void  *aster_xcalloc(size_t count, size_t size);
void  *aster_xrealloc(void *ptr, size_t bytes);

/* ---- files ------------------------------------------------------------- */
/* Read a whole file. Returns NULL and sets *out_ok = 0 when the file cannot
 * be opened. The buffer is NUL-terminated for convenience. */
char  *aster_read_file(const char *path, size_t *out_len, int *out_ok);

/* Write via "<path>.tmp" then an atomic rename, so an interrupted write can
 * never corrupt the previous good file. */
void   aster_write_file_atomic(const char *path, const void *data, size_t len);

void   aster_mkdir_for_file(const char *path);  /* mkdir -p on the dirname   */
int    aster_file_exists(const char *path);

/* ---- hashing ------------------------------------------------------------ */
uint32_t aster_crc32(const void *data, size_t len);
void     aster_sha256_hex(const void *data, size_t len, char out[65]);

/* ---- time and randomness ----------------------------------------------- */
double   aster_now_seconds(void);   /* monotonic seconds since some origin */
uint32_t aster_rng_u32(uint32_t *state);            /* xorshift32, seeded  */
float    aster_rng_uniform(uint32_t *state);        /* [0,1)               */
float    aster_rng_normal(uint32_t *state);         /* Box-Muller          */
void     aster_rng_seed_init(uint32_t *state, uint32_t seed);

#endif /* ASTER_UTIL_H */
