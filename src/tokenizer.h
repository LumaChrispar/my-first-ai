/* tokenizer.h - byte-level tokenizer with reserved conversation markers.
 *
 * Token IDs:
 *   0..255  one raw byte of the input (so UTF-8 text becomes its UTF-8 bytes)
 *   256 SYSTEM      257 USER      258 ASSISTANT      259 END
 *   260 PAD         (batching only; padding never contributes to the loss)
 *
 * There is no third-party BPE here on purpose: the first milestone must be
 * inspectable and must not download anything. Byte tokenization is simple but
 * wasteful, so 128 tokens of context is far less than 128 words.
 */
#ifndef ASTER_TOKENIZER_H
#define ASTER_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define ASTER_VOCAB            261
#define ASTER_BYTE_MAX         255
#define TOK_SYSTEM             256
#define TOK_USER               257
#define TOK_ASSISTANT          258
#define TOK_END                259
#define TOK_PAD                260
#define ASTER_TOKENIZER_VERSION 1

typedef struct { uint16_t *ids; size_t count; size_t cap; int overflowed; } TokenList;

void token_list_init(TokenList *tl);
void token_list_free(TokenList *tl);
void token_list_push(TokenList *tl, uint16_t id);
void token_list_push_many(TokenList *tl, const uint16_t *ids, size_t n);

/* Encode UTF-8 text to token IDs. Never reads past len; sets
 * tl->overflowed (instead of writing out of bounds) if cap is too small.
 * Returns 0 on success, -1 if the output capacity was insufficient. */
int  tok_encode(const char *text, size_t len, TokenList *out);

/* Decode token IDs back to bytes. Byte tokens 0..255 are emitted verbatim;
 * SYSTEM/USER/ASSISTANT/PAD are skipped; END becomes a newline when
 * end_marker_newline is non-zero. `out` is always NUL-terminated. */
size_t tok_decode(const uint16_t *ids, size_t n, char *out, size_t cap, int end_marker_newline);

/* Replace invalid UTF-8 byte sequences with U+FFFD so malformed model output
 * can never crash or corrupt the UI. Writes at most cap bytes, NUL-terminated.
 * Returns 1 if any replacement was needed. */
int utf8_sanitize(const char *in, size_t len, char *out, size_t cap);

/* Human-readable name for a token id (for debugging only). */
const char *tok_name(uint16_t id);

/* 1 if the text is plausible UTF-8 (used by the selftest). */
int utf8_is_valid(const char *in, size_t len);

#endif /* ASTER_TOKENIZER_H */
