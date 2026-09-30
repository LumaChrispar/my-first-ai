/* tokenizer.h - sub-word (BPE) tokenizer with byte fallback.
 *
 * Token IDs:
 *   0..255  one raw byte of the input (so UTF-8 text becomes its UTF-8 bytes)
 *   256 SYSTEM      257 USER      258 ASSISTANT      259 END
 *   260 PAD         (batching only; padding never contributes to the loss)
 *   261..  learned merges, in the order they were learned. A LOWER id was
 *          learned EARLIER, and earlier merges win when two of them could
 *          both apply to the same pair of symbols.
 *
 * The 256 byte ids are always present, which is what makes byte fallback
 * work: any byte sequence is representable, so there is no unknown token and
 * no input the encoder can reject. A vocabulary with zero merges is therefore
 * a valid vocabulary and behaves exactly like the old byte-level tokenizer.
 *
 * Everything the decoder and the encoder need is DERIVED from the merge list
 * (tok_len, tok_bytes, rank) and rebuilt on load. Only the merge pairs are
 * ever stored, so there is exactly one source of truth and no way for the
 * derived tables to disagree with it.
 */
#ifndef ASTER_TOKENIZER_H
#define ASTER_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define ASTER_BYTE_MAX         255
#define TOK_SYSTEM             256
#define TOK_USER               257
#define TOK_ASSISTANT          258
#define TOK_END                259
#define TOK_PAD                260

/* First id available to a learned merge. */
#define TOK_FIRST_MERGE        261
/* Vocab with no merges: pure byte fallback. */
#define ASTER_BASE_VOCAB       261
/* Caps. The merge table is fixed-size so it can live inside a checkpoint
 * header and inside AsterModel without a second allocation. */
#define ASTER_MAX_MERGES       1024
#define ASTER_MAX_VOCAB        (ASTER_BASE_VOCAB + ASTER_MAX_MERGES)
#define ASTER_DEFAULT_MERGES   1024

/* Ceiling on how many bytes one token may expand to. A corpus with almost no
 * whitespace ("aaaaaaa...") would otherwise let a merge chain grow a token
 * hundreds of bytes long, which is legal but useless: it cannot be generated
 * inside a 128-token context and it silently defeats the prompt budget.
 * Learning refuses to create one, and loading rejects a table containing one,
 * both loudly. */
#define ASTER_BPE_MAX_TOK_BYTES 128

#define ASTER_TOKENIZER_VERSION 2

typedef struct { uint16_t *ids; size_t count; size_t cap; int overflowed; } TokenList;

/* The learned vocabulary. Large enough to be stored by value; the only
 * separately allocated members are tok_bytes and the optional rank index. */
typedef struct {
    int      n_merges;
    uint16_t merge_a[ASTER_MAX_MERGES];
    uint16_t merge_b[ASTER_MAX_MERGES];
    /* Derived. tok_len[i] is how many bytes token i expands to; the bytes
     * themselves live contiguously in tok_bytes at offset sum(tok_len[0..i)).
     * Structural markers have length 0 and are never decoded. */
    uint16_t tok_len[ASTER_MAX_VOCAB];
    uint8_t *tok_bytes;
    size_t   tok_bytes_n;
    /* Derived, lazily built by aster_vocab_build_index(). Read-only once
     * built, so copies of a struct may share the pointer safely.
     * rank[a * vocab + b] is the merge id that joins a then b, or -1. */
    int16_t *rank;
    int      rank_vocab;
    /* Bytes per token measured over the corpus the merges were learned from.
     * Used only to give the UI an honest estimate; never used to enforce a
     * limit, because an estimate that silently truncated a prompt would be
     * worse than no estimate at all. */
    double   bytes_per_token;
} AsterVocab;

void token_list_init(TokenList *tl);
void token_list_free(TokenList *tl);
void token_list_push(TokenList *tl, uint16_t id);
void token_list_push_many(TokenList *tl, const uint16_t *ids, size_t n);

/* ---- vocabulary --------------------------------------------------------- */

/* Sets up a zero-merge vocabulary: valid, and identical in behaviour to the
 * old byte tokenizer. */
void aster_vocab_init(AsterVocab *v);

void aster_vocab_free(AsterVocab *v);

/* Number of token ids this vocabulary defines. */
int  aster_vocab_size(const AsterVocab *v);

/* 1 if id is a structural marker (SYSTEM/USER/ASSISTANT/END/PAD). */
int  tok_is_structural(uint16_t id);

/* Rebuilds tok_len/tok_bytes from the merge pairs, then (if asked) the rank
 * index. Must be called after merge_a/merge_b are set and before anything
 * encodes or decodes. Returns 0, or -1 with a reason in err if the merge
 * table is malformed (a merge that references a token that did not exist
 * when it was learned, a self-merge, an out-of-range id). */
int  aster_vocab_finalize(AsterVocab *v, int build_index, char *err, size_t errlen);

/* Builds only the rank index. Safe to call more than once. */
int  aster_vocab_build_index(AsterVocab *v);

/* The bytes a token expands to. Returns 0 on success, -1 for a structural
 * marker or an unknown id. */
int  tok_spelling(const AsterVocab *v, uint16_t id, const char **out, size_t *len);

/* Human-readable name for a token id (for debugging only). Static storage;
 * not reentrant. Merge ids report the bytes they spell. */
const char *tok_name(const AsterVocab *v, uint16_t id);

/* ---- encode / decode ---------------------------------------------------- */

/* The largest number of bytes any one token expands to. Generation sizes its
 * output buffer from this: a single token can spell a whole word, so the old
 * "one token, one byte" sizing would truncate every reply. */
int  aster_vocab_max_token_bytes(const AsterVocab *v);

/* Encode UTF-8 text to token IDs using the learned merges. The round-trip
 * property
 *     decode(encode(s)) == s
 * holds for every s, because the 256 byte ids are always available.
 * Returns 0 on success, -1 if the vocabulary has not been finalized, and -2
 * if the text contains a NUL -- which is an error, not something to skip,
 * because skipping it would break the property above. */
int  tok_encode(const AsterVocab *v, const char *text, size_t len, TokenList *out);

/* Decode token IDs back to bytes. SYSTEM/USER/ASSISTANT/PAD are skipped;
 * END becomes a newline when end_marker_newline is non-zero. `out` is always
 * NUL-terminated and the return value is what was written, NOT the number of
 * input ids: one id can spell many bytes. */
size_t tok_decode(const AsterVocab *v, const uint16_t *ids, size_t n,
                  char *out, size_t cap, int end_marker_newline);

/* Replace invalid UTF-8 byte sequences with U+FFFD so malformed model output
 * can never crash or corrupt the UI. Writes at most cap bytes, NUL-terminated.
 * Returns 1 if any replacement was needed. */
int utf8_sanitize(const char *in, size_t len, char *out, size_t cap);

/* 1 if the text is plausible UTF-8 (used by the selftest). */
int utf8_is_valid(const char *in, size_t len);

/* ---- learning merges ----------------------------------------------------
 * Kept behind an opaque handle so the word table and the pair-count table
 * stay out of this header. Text is fed in by the caller rather than read
 * from a path, so the learner never touches the filesystem and can only
 * ever see the split the caller chose to give it.
 *
 * This matters: the merges must be learned from the TRAINING split alone.
 * A vocabulary fitted on the validation set leaks it, and the held-out loss
 * stops meaning anything. */
typedef struct AsterVocabLearner AsterVocabLearner;

AsterVocabLearner *bpe_learner_new(int max_merges);
void bpe_learner_free(AsterVocabLearner *L);

/* Adds raw text. Words are split on whitespace, with the delimiter attached
 * to the front of the following word, so " the" is learnable as one unit. */
void bpe_learner_add_text(AsterVocabLearner *L, const char *text, size_t len);

/* Runs the merge loop and fills *out. Returns 0, or -1 with a reason in err.
 * Learning is deterministic: the same text in the same order always yields
 * the same merge table, because the best pair is chosen by count and then by
 * the lowest (a, b) rather than by whatever the hash table happens to hold
 * first. */
int bpe_learner_finish(AsterVocabLearner *L, AsterVocab *out, char *err, size_t errlen);

/* Words seen, distinct words kept, and the merges actually learned. */
void bpe_learner_stats(const AsterVocabLearner *L, size_t *words, size_t *distinct, int *merges);

/* ---- vocabulary files ----------------------------------------------------
 * A standalone .vocab file, so a vocabulary can be inspected, diffed, and
 * reused across training runs. It is a convenience, not the source of truth:
 * the checkpoint carries its own copy of the merge table, and that is the one
 * the model uses. A checkpoint and a .vocab file can disagree without corrupting
 * anything, which is the point -- but it does mean the file is worth
 * regenerating rather than hand-editing.
 *
 * Format: "ASTERVB1" + u32 n_merges + f64 bytes_per_token + u16 pairs.
 * A zero-merge vocabulary is a valid file, so byte-fallback behaviour can be
 * saved and shared like any other. */
int  aster_vocab_save(const AsterVocab *v, const char *path, char *err, size_t errlen);
AsterVocab *aster_vocab_load(const char *path, char *err, size_t errlen);

#endif /* ASTER_TOKENIZER_H */
