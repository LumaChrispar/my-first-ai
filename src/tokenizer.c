#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================ token list == */

void token_list_init(TokenList *tl) {
    tl->ids = NULL; tl->count = 0; tl->cap = 0; tl->overflowed = 0;
}

void token_list_free(TokenList *tl) {
    free(tl->ids);
    token_list_init(tl);
}

void token_list_push(TokenList *tl, uint16_t id) {
    if (tl->count == tl->cap) {
        tl->cap = tl->cap ? tl->cap * 2 : 256;
        tl->ids = (uint16_t *)aster_xrealloc(tl->ids, aster_size_mul(tl->cap, sizeof(uint16_t)));
    }
    tl->ids[tl->count++] = id;
}

void token_list_push_many(TokenList *tl, const uint16_t *ids, size_t n) {
    for (size_t i = 0; i < n; ++i) token_list_push(tl, ids[i]);
}

/* ============================================================ vocabulary == */

int aster_vocab_size(const AsterVocab *v) {
    return ASTER_BASE_VOCAB + (v ? v->n_merges : 0);
}

int tok_is_structural(uint16_t id) {
    return id >= TOK_SYSTEM && id <= TOK_PAD;
}

void aster_vocab_init(AsterVocab *v) {
    memset(v, 0, sizeof *v);
    v->n_merges = 0;
    v->bytes_per_token = 1.0;
    v->rank = NULL;
    v->rank_vocab = 0;
    /* A zero-merge vocabulary is valid and behaves exactly like the old byte
     * tokenizer, which makes it the natural regression baseline. */
    v->tok_bytes = (uint8_t *)aster_xmalloc(ASTER_BYTE_MAX + 1);
    for (int i = 0; i <= ASTER_BYTE_MAX; ++i) {
        v->tok_len[i] = 1;
        v->tok_bytes[i] = (uint8_t)i;
    }
    v->tok_bytes_n = (size_t)ASTER_BYTE_MAX + 1;
}

void aster_vocab_free(AsterVocab *v) {
    if (!v) return;
    free(v->tok_bytes);
    free(v->rank);
    v->tok_bytes = NULL;
    v->rank = NULL;
    v->tok_bytes_n = 0;
    v->rank_vocab = 0;
}

int aster_vocab_build_index(AsterVocab *v) {
    const int V = aster_vocab_size(v);
    if (v->rank && v->rank_vocab == V) return 0;
    free(v->rank);
    v->rank = NULL;
    /* One entry per ordered token pair. At the 1024-merge cap this is 3.3 MB,
     * which is a fine price for turning encode into a table lookup instead of
     * a hash probe on the hot path. */
    v->rank = (int16_t *)aster_xmalloc(aster_size_mul((size_t)V * (size_t)V, sizeof(int16_t)));
    memset(v->rank, 0xFF, (size_t)V * (size_t)V * sizeof(int16_t));   /* -1 = no merge */
    for (int i = 0; i < v->n_merges; ++i) {
        const int a = v->merge_a[i], b = v->merge_b[i];
        v->rank[(size_t)a * (size_t)V + (size_t)b] = (int16_t)i;
    }
    v->rank_vocab = V;
    return 0;
}

int aster_vocab_finalize(AsterVocab *v, int build_index, char *err, size_t errlen) {
#define BADF(...) do { if (err && errlen) snprintf(err, errlen, __VA_ARGS__); return -1; } while (0)
    if (!v) BADF("no vocabulary");
    if (v->n_merges < 0 || v->n_merges > ASTER_MAX_MERGES)
        BADF("merge count %d is outside 0..%d", v->n_merges, ASTER_MAX_MERGES);

    /* A merge may only join symbols that already existed when it was learned,
     * and may never contain a structural marker. Checking the id against
     * TOK_FIRST_MERGE + i does both at once: ids 0..255 pass as raw bytes,
     * ids 261..260+i pass as earlier merges, and 256..260 fall in the gap and
     * are rejected. A merge swallowing END would erase a turn boundary on
     * decode, which is the kind of bug that shows up as a model that quietly
     * learns to stop mid-sentence.
     *
     * a == b IS allowed. Joining a token to itself is how a repeated run
     * compresses -- "aaaaaa" becomes one token by merging (X, X) -- and it is
     * bounded by the token-length cap below, since each such merge doubles
     * the length. */
    for (int i = 0; i < v->n_merges; ++i) {
        const int a = v->merge_a[i], b = v->merge_b[i];
        if (!((a >= 0 && a <= ASTER_BYTE_MAX) || (a >= TOK_FIRST_MERGE && a < TOK_FIRST_MERGE + i)))
            BADF("merge %d references token %d, which did not exist when it was learned "
                 "(or is a reserved marker)", i, a);
        if (!((b >= 0 && b <= ASTER_BYTE_MAX) || (b >= TOK_FIRST_MERGE && b < TOK_FIRST_MERGE + i)))
            BADF("merge %d references token %d, which did not exist when it was learned "
                 "(or is a reserved marker)", i, b);
    }

    /* Rebuild tok_len from scratch. A merge's byte length is the sum of its
     * two halves, so the whole table falls out of the merge list alone. */
    for (int i = 0; i <= ASTER_BYTE_MAX; ++i) v->tok_len[i] = 1;
    for (int i = TOK_SYSTEM; i < TOK_FIRST_MERGE; ++i) v->tok_len[i] = 0;  /* markers spell nothing */

    for (int i = 0; i < v->n_merges; ++i) {
        const int id = TOK_FIRST_MERGE + i;
        const int len = (int)v->tok_len[v->merge_a[i]] + (int)v->tok_len[v->merge_b[i]];
        if (len > ASTER_BPE_MAX_TOK_BYTES)
            BADF("merge %d would create a %d-byte token, over the %d-byte limit; "
                 "this usually means the corpus has very few word boundaries",
                 i, len, ASTER_BPE_MAX_TOK_BYTES);
        if (len == 0) BADF("merge %d has zero length", i);
        v->tok_len[id] = (uint16_t)len;
    }

    size_t total = 0;
    for (int id = 0; id < aster_vocab_size(v); ++id) total += v->tok_len[id];
    free(v->tok_bytes);
    v->tok_bytes = (uint8_t *)aster_xmalloc(total ? total : 1);
    v->tok_bytes_n = total;

    /* Lay the bytes down in id order, remembering where each token starts.
     * A merge's two halves are always lower ids, so their bytes are already in
     * the table by the time the merge is written. */
    size_t off[ASTER_MAX_VOCAB];
    size_t at = 0;
    for (int i = 0; i <= ASTER_BYTE_MAX; ++i) { off[i] = at; v->tok_bytes[at++] = (uint8_t)i; }
    for (int i = 0; i < v->n_merges; ++i) {
        const int id = TOK_FIRST_MERGE + i;
        const int a = v->merge_a[i], b = v->merge_b[i];
        off[id] = at;
        memcpy(v->tok_bytes + at, v->tok_bytes + off[a], v->tok_len[a]);
        at += v->tok_len[a];
        memcpy(v->tok_bytes + at, v->tok_bytes + off[b], v->tok_len[b]);
        at += v->tok_len[b];
    }
    if (at != total) BADF("internal: token byte table is %zu bytes, expected %zu", at, total);

    if (build_index && aster_vocab_build_index(v) != 0) BADF("could not build the merge index");
    return 0;
#undef BADF
}

int aster_vocab_max_token_bytes(const AsterVocab *v) {
    int m = 1;
    if (!v) return m;
    for (int i = TOK_FIRST_MERGE; i < aster_vocab_size(v); ++i)
        if ((int)v->tok_len[i] > m) m = (int)v->tok_len[i];
    return m;
}

int tok_spelling(const AsterVocab *v, uint16_t id, const char **out, size_t *len) {    if (!v || !v->tok_bytes || id >= (uint16_t)aster_vocab_size(v)) return -1;
    if (tok_is_structural(id)) return -1;
    size_t at = 0;
    for (int i = 0; i < (int)id; ++i) at += v->tok_len[i];
    if (out) *out = (const char *)v->tok_bytes + at;
    if (len) *len = v->tok_len[id];
    return 0;
}

const char *tok_name(const AsterVocab *v, uint16_t id) {
    static char buf[ASTER_BPE_MAX_TOK_BYTES * 4 + 32];
    static const char *names[] = {"SYSTEM", "USER", "ASSISTANT", "END", "PAD"};
    if (id <= ASTER_BYTE_MAX) return "byte";
    if (id >= TOK_SYSTEM && id <= TOK_PAD) return names[id - TOK_SYSTEM];
    const char *s = NULL;
    size_t n = 0;
    if (tok_spelling(v, id, &s, &n) != 0) return "invalid";
    size_t j = 0;
    for (size_t i = 0; i < n && j + 5 < sizeof buf; ++i) {
        const unsigned char c = (unsigned char)s[i];
        if (c >= 0x20 && c < 0x7F) buf[j++] = (char)c;
        else j += (size_t)snprintf(buf + j, sizeof buf - j, "\\x%02X", c);
    }
    buf[j] = '\0';
    return buf;
}

/* ============================================================ encode/decode */

/* Encode UTF-8 text to token IDs using the learned merges.
 *
 * Each pass finds the adjacent pair whose merge was learned EARLIEST (lowest
 * id, which is the strongest claim) and merges every occurrence of that one
 * pair, then rescans. Rescanning is what makes this simple rather than fast:
 * the textbook version keeps a priority queue of candidate positions, and the
 * queue is where the bugs live. A rescan is O(n) with no bookkeeping to get
 * wrong, and the worst case is O(n^2) on a field of n bytes -- irrelevant for
 * a prompt, and a one-time cost of seconds for a whole training corpus.
 *
 * Returns 0 on success, -1 if the vocabulary was not finalized, and -2 if the
 * text contains a NUL. A NUL is an ERROR rather than something to skip: the
 * byte ids cover every value except zero, so silently dropping one would make
 * decode(encode(s)) != s, and a tokenizer that quietly loses data is worse
 * than one that stops. Data cleaning strips NULs long before here, so
 * reaching this means something upstream is already wrong.
 */
int tok_encode(const AsterVocab *v, const char *text, size_t len, TokenList *out) {
    out->overflowed = 0;
    if (!v) return -1;
    if (v->n_merges > 0 && !v->rank) return -1;   /* not finalized; fail loudly */

    for (size_t i = 0; i < len; ++i)
        if (text[i] == '\0') return -2;

    uint16_t *s = (uint16_t *)aster_xmalloc((len ? len : 1) * sizeof(uint16_t));
    size_t n = 0;
    for (size_t i = 0; i < len; ++i) s[n++] = (uint16_t)(unsigned char)text[i];

    const int V = aster_vocab_size(v);
    const int16_t *rank = v->rank;
    uint16_t *tmp = (uint16_t *)aster_xmalloc((n ? n : 1) * sizeof(uint16_t));

    for (;;) {
        int best = -1;
        for (size_t i = 0; i + 1 < n; ++i) {
            const int r = rank[(size_t)s[i] * (size_t)V + (size_t)s[i + 1]];
            if (r >= 0 && (best < 0 || r < best)) best = r;
        }
        if (best < 0) break;
        const uint16_t a = v->merge_a[best], b = v->merge_b[best];
        const uint16_t merged = (uint16_t)(TOK_FIRST_MERGE + best);
        size_t m = 0;
        for (size_t i = 0; i < n; ) {
            if (i + 1 < n && s[i] == a && s[i + 1] == b) { tmp[m++] = merged; i += 2; }
            else tmp[m++] = s[i++];
        }
        n = m;
        memcpy(s, tmp, n * sizeof(uint16_t));
    }

    for (size_t i = 0; i < n; ++i) token_list_push(out, s[i]);
    free(tmp);
    free(s);
    return out->overflowed ? -1 : 0;
}

size_t tok_decode(const AsterVocab *v, const uint16_t *ids, size_t n,
                  char *out, size_t cap, int end_marker_newline) {
    if (cap == 0) return 0;
    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
        if (ids[i] == TOK_END) {
            if (!end_marker_newline) continue;
            if (j + 1 < cap) out[j++] = '\n';
            continue;
        }
        if (tok_is_structural(ids[i])) continue;   /* SYSTEM/USER/ASSISTANT/PAD */
        const char *s = NULL;
        size_t len = 0;
        if (tok_spelling(v, ids[i], &s, &len) != 0) continue;
        for (size_t k = 0; k < len && j + 1 < cap; ++k) out[j++] = s[k];
    }
    out[j] = '\0';
    return j;
}

/* ================================================================ utf-8 === */
/* Length of the UTF-8 sequence starting at in[0], or 0 if it is not a valid
 * lead byte. Rejects overlong forms, surrogates, and out-of-range code points
 * so that sanitize() is actually correct rather than merely tolerant. */
static int utf8_seq_len(const unsigned char *in) {
    unsigned char c = in[0];
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) return 2;
    if (c >= 0xE0 && c <= 0xEF) return 3;
    if (c >= 0xF0 && c <= 0xF4) return 4;
    return 0;
}

static int utf8_seq_valid(const unsigned char *in, int len) {
    for (int k = 1; k < len; ++k) if ((in[k] & 0xC0) != 0x80) return 0;
    if (len == 2) return 1;
    if (len == 3) {
        if (in[0] == 0xE0 && in[1] < 0xA0) return 0;          /* overlong */
        if (in[0] == 0xED && in[1] >= 0xA0) return 0;         /* surrogate */
        return 1;
    }
    if (in[0] == 0xF0 && in[1] < 0x90) return 0;              /* overlong */
    if (in[0] == 0xF4 && in[1] >= 0x90) return 0;             /* > U+10FFFF */
    return 1;
}

int utf8_sanitize(const char *in, size_t len, char *out, size_t cap) {
    if (cap == 0) return 0;
    const unsigned char *p = (const unsigned char *)in;
    size_t i = 0, j = 0;
    int replaced = 0;
    while (i < len) {
        int sl = utf8_seq_len(p + i);
        if (sl > 0 && (i + (size_t)sl <= len) && utf8_seq_valid(p + i, sl)) {
            if (j + (size_t)sl + 1 < cap) { memcpy(out + j, p + i, (size_t)sl); j += (size_t)sl; }
            i += (size_t)sl;
        } else {
            replaced = 1;
            if (j + 4 < cap) { memcpy(out + j, "\xEF\xBF\xBD", 3); j += 3; }
            i += 1;
        }
    }
    out[j] = '\0';
    return replaced;
}

int utf8_is_valid(const char *in, size_t len) {
    const unsigned char *p = (const unsigned char *)in;
    size_t i = 0;
    while (i < len) {
        int sl = utf8_seq_len(p + i);
        if (sl == 0 || i + (size_t)sl > len || !utf8_seq_valid(p + i, sl)) return 0;
        i += (size_t)sl;
    }
    return 1;
}

/* ================================================================ learning ==
 *
 * A word table (open-addressed) plus a DENSE pair-count table.
 *
 * The pair table is a flat V-by-V array rather than a hash. V is at most
 * 1285, so the whole table is 13 MB and every lookup is one multiply and one
 * load -- no probing, no collisions, nothing to get wrong. Choosing the next
 * merge becomes a straight linear walk for the maximum, which is the
 * operation the loop performs 1024 times.
 *
 * A hash table would be asymptotically nicer and practically no better here,
 * and it would add the one genuinely fiddly part of the job: keeping counts
 * correct across a table that grows while it is being decremented.
 *
 * Words are stored once with a frequency, and pair counts are weighted by
 * that frequency. The loop therefore rescans only the DISTINCT words, not the
 * corpus: a 20 MB corpus is a few hundred thousand distinct words, so the
 * work is bounded by vocabulary size rather than by corpus size.
 */

#define BPE_MAX_WORDS      200000u
#define BPE_MAX_WORD_LEN   64
#define BPE_MIN_PAIR_COUNT 2

typedef struct { uint16_t *syms; uint16_t n; uint16_t bytes; uint32_t count; } BpeWord;

struct AsterVocabLearner {
    int max_merges;

    BpeWord  *w;
    size_t    nw, cw;
    uint32_t *wslot;          /* word index + 1; 0 means empty */
    size_t    wslot_cap;
    size_t    words_seen;

    int64_t  *pair;           /* [a * pv + b], weighted occurrences */
    int       pv;             /* fixed at TOK_FIRST_MERGE + max_merges */

    uint16_t  tok_len[ASTER_MAX_VOCAB];
    int       n_merges;
};

static uint64_t fnv1a64(const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

AsterVocabLearner *bpe_learner_new(int max_merges) {
    if (max_merges < 0) max_merges = 0;
    if (max_merges > ASTER_MAX_MERGES) max_merges = ASTER_MAX_MERGES;
    AsterVocabLearner *L = (AsterVocabLearner *)aster_xcalloc(1, sizeof *L);
    L->max_merges = max_merges;
    L->wslot_cap = 1u << 16;
    L->wslot = (uint32_t *)aster_xcalloc(L->wslot_cap, sizeof(uint32_t));
    L->pv = TOK_FIRST_MERGE + max_merges;
    L->pair = (int64_t *)aster_xcalloc((size_t)L->pv * (size_t)L->pv, sizeof(int64_t));
    for (int i = 0; i <= ASTER_BYTE_MAX; ++i) L->tok_len[i] = 1;
    return L;
}

void bpe_learner_free(AsterVocabLearner *L) {
    if (!L) return;
    for (size_t i = 0; i < L->nw; ++i) free(L->w[i].syms);
    free(L->w); free(L->wslot); free(L->pair);
    free(L);
}

void bpe_learner_stats(const AsterVocabLearner *L, size_t *words, size_t *distinct, int *merges) {
    if (!L) return;
    if (words) *words = L->words_seen;
    if (distinct) *distinct = L->nw;
    if (merges) *merges = L->n_merges;
}

/* ---- the word table ---- */

static void word_table_grow(AsterVocabLearner *L) {
    const size_t nc = L->wslot_cap * 2;
    uint32_t *ns = (uint32_t *)aster_xcalloc(nc, sizeof(uint32_t));
    for (size_t i = 0; i < L->wslot_cap; ++i) {
        const uint32_t v = L->wslot[i];
        if (!v) continue;
        const BpeWord *w = &L->w[v - 1];
        size_t p = (size_t)fnv1a64(w->syms, (size_t)w->n * sizeof(uint16_t)) & (nc - 1);
        while (ns[p]) p = (p + 1) & (nc - 1);
        ns[p] = v;
    }
    free(L->wslot);
    L->wslot = ns;
    L->wslot_cap = nc;
}

static void learner_add_word(AsterVocabLearner *L, const char *s, size_t len) {
    L->words_seen++;
    if (len == 0 || len > BPE_MAX_WORD_LEN) return;
    if (L->nw >= BPE_MAX_WORDS) return;   /* rare words still encode, via bytes */

    size_t p = (size_t)fnv1a64(s, len) & (L->wslot_cap - 1);
    while (L->wslot[p]) {
        BpeWord *w = &L->w[L->wslot[p] - 1];
        if (w->bytes == len && memcmp(w->syms, s, len) == 0) { w->count++; return; }
        p = (p + 1) & (L->wslot_cap - 1);
    }
    if (L->nw + 1 > L->cw) {
        L->cw = L->cw ? L->cw * 2 : 1024;
        L->w = (BpeWord *)aster_xrealloc(L->w, aster_size_mul(L->cw, sizeof(BpeWord)));
    }
    BpeWord *w = &L->w[L->nw];
    w->syms = (uint16_t *)aster_xmalloc(len * sizeof(uint16_t));
    for (size_t i = 0; i < len; ++i) w->syms[i] = (uint16_t)(unsigned char)s[i];
    w->n = (uint16_t)len;
    w->bytes = (uint16_t)len;
    w->count = 1;
    L->wslot[p] = (uint32_t)L->nw + 1;
    L->nw++;
    if (L->nw * 10 > L->wslot_cap * 6) word_table_grow(L);
}

void bpe_learner_add_text(AsterVocabLearner *L, const char *text, size_t len) {
    if (!L || !text) return;
    size_t i = 0;
    while (i < len) {
        const size_t start = i;
        /* Take one leading delimiter, then a run of non-delimiters. So
         * "hello world" yields "hello" and " world": the space rides along
         * with the word it introduces, which is what lets " the" be learned
         * as a single token. Splitting on an ASCII byte cannot land inside a
         * UTF-8 sequence, because continuation bytes are all >= 0x80. */
        if (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t') i++;
        while (i < len && !(text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t')) i++;
        if (i > start) learner_add_word(L, text + start, i - start);
    }
}

int bpe_learner_finish(AsterVocabLearner *L, AsterVocab *out, char *err, size_t errlen) {
#define BADF(...) do { if (err && errlen) snprintf(err, errlen, __VA_ARGS__); return -1; } while (0)
    if (!L || !out) BADF("no learner");
    if (L->nw == 0) BADF("no words were collected, so no merges could be learned");

    /* Every word starts as raw bytes and every merge is built from symbols
     * that already exist, so ids 256..260 are structurally unreachable here
     * and need no check. aster_vocab_finalize re-validates anyway, because
     * the loader can be handed a file that never came from this function. */
    for (size_t i = 0; i < L->nw; ++i) {
        const BpeWord *w = &L->w[i];
        for (uint16_t k = 0; k + 1 < w->n; ++k)
            L->pair[(size_t)w->syms[k] * (size_t)L->pv + (size_t)w->syms[k + 1]] += w->count;
    }

    uint16_t *orig = (uint16_t *)aster_xmalloc(BPE_MAX_WORD_LEN * sizeof(uint16_t));
    uint16_t *next = (uint16_t *)aster_xmalloc(BPE_MAX_WORD_LEN * sizeof(uint16_t));

    while (L->n_merges < L->max_merges) {
        /* Best pair = highest count; ties broken by the lowest (a, b), which
         * the row-major walk below gets for free. So the result never depends
         * on anything but the counts, and learning is reproducible. */
        int64_t best = -1;
        int ba = -1, bb = -1;
        for (int a = 0; a < L->pv && best < L->n_merges + TOK_FIRST_MERGE; ++a) {
            const int64_t *row = L->pair + (size_t)a * (size_t)L->pv;
            for (int b = 0; b < L->pv; ++b)
                if (row[b] > BPE_MIN_PAIR_COUNT && row[b] > best) { best = row[b]; ba = a; bb = b; }
        }
        if (ba < 0) break;

        const int id = TOK_FIRST_MERGE + L->n_merges;
        const int newlen = (int)L->tok_len[ba] + (int)L->tok_len[bb];

        if (newlen > ASTER_BPE_MAX_TOK_BYTES) {
            /* Too long to be useful. Retire this pair and try the next rather
             * than stopping: one over-long frequent pair does not mean the
             * rest of the corpus is degenerate. */
            L->pair[(size_t)ba * (size_t)L->pv + (size_t)bb] = 0;
            continue;
        }
        L->tok_len[id] = (uint16_t)newlen;

        for (size_t wi = 0; wi < L->nw; ++wi) {
            BpeWord *w = &L->w[wi];
            if (w->n < 2) continue;
            memcpy(orig, w->syms, (size_t)w->n * sizeof(uint16_t));
            uint16_t m = 0;
            for (uint16_t k = 0; k < w->n; ) {
                if (k + 1 < w->n && w->syms[k] == ba && w->syms[k + 1] == bb) { next[m++] = (uint16_t)id; k += 2; }
                else next[m++] = w->syms[k++];
            }
            if (m == w->n) continue;                  /* nothing changed */
            for (uint16_t k = 0; k + 1 < w->n; ++k)
                L->pair[(size_t)orig[k] * (size_t)L->pv + (size_t)orig[k + 1]] -= w->count;
            for (uint16_t k = 0; k + 1 < m; ++k)
                L->pair[(size_t)next[k] * (size_t)L->pv + (size_t)next[k + 1]] += w->count;
            memcpy(w->syms, next, (size_t)m * sizeof(uint16_t));
            w->n = m;
        }

        out->merge_a[L->n_merges] = (uint16_t)ba;
        out->merge_b[L->n_merges] = (uint16_t)bb;
        L->n_merges++;
    }

    /* Bytes per token, measured on what was actually learned. This is only an
     * estimate for the UI's character counter; the server enforces the real
     * limit in token space. */
    {
        double bytes = 0.0, toks = 0.0;
        for (size_t i = 0; i < L->nw; ++i) {
            bytes += (double)L->w[i].bytes * (double)L->w[i].count;
            toks  += (double)L->w[i].n * (double)L->w[i].count;
        }
        out->bytes_per_token = (toks > 0.0) ? bytes / toks : 1.0;
    }

    free(orig);
    free(next);

    out->n_merges = L->n_merges;
    if (aster_vocab_finalize(out, 1, err, errlen) != 0) return -1;
    return 0;
#undef BADF
}

/* ======================================================= vocabulary files == */

#define VOCAB_MAGIC "ASTERVB1"
#define VOCAB_HEAD  (8 + 4 + 8)

static void put_u32le(unsigned char *p, uint32_t x) {
    p[0] = (unsigned char)(x);       p[1] = (unsigned char)(x >> 8);
    p[2] = (unsigned char)(x >> 16); p[3] = (unsigned char)(x >> 24);
}

static uint32_t get_u32le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_f64le(unsigned char *p, double d) {
    uint64_t bits;
    memcpy(&bits, &d, 8);
    for (int i = 0; i < 8; ++i) p[i] = (unsigned char)(bits >> (8 * i));
}

static double get_f64le(const unsigned char *p) {
    uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) bits |= (uint64_t)p[i] << (8 * i);
    double d;
    memcpy(&d, &bits, 8);
    return d;
}

int aster_vocab_save(const AsterVocab *v, const char *path, char *err, size_t errlen) {
#define BADF(...) do { if (err && errlen) snprintf(err, errlen, __VA_ARGS__); return -1; } while (0)
    if (!v) BADF("no vocabulary to save");
    if (!path) BADF("no output path");
    if (v->n_merges < 0 || v->n_merges > ASTER_MAX_MERGES)
        BADF("merge count %d is out of range (0..%d)", v->n_merges, ASTER_MAX_MERGES);

    size_t n = VOCAB_HEAD + 4u * (size_t)v->n_merges;
    unsigned char *buf = (unsigned char *)aster_xcalloc(n, 1);
    memcpy(buf, VOCAB_MAGIC, 8);
    put_u32le(buf + 8, (uint32_t)v->n_merges);
    put_f64le(buf + 12, v->bytes_per_token);
    for (int i = 0; i < v->n_merges; ++i) {
        unsigned char *p = buf + VOCAB_HEAD + 4 * (size_t)i;
        p[0] = (unsigned char)(v->merge_a[i]);       p[1] = (unsigned char)(v->merge_a[i] >> 8);
        p[2] = (unsigned char)(v->merge_b[i]);       p[3] = (unsigned char)(v->merge_b[i] >> 8);
    }
    aster_write_file_atomic(path, buf, n);
    free(buf);
    return 0;
#undef BADF
}

AsterVocab *aster_vocab_load(const char *path, char *err, size_t errlen) {
#define BADF(...) do { if (err && errlen) snprintf(err, errlen, __VA_ARGS__); return NULL; } while (0)
    if (!path) BADF("no vocabulary path");
    size_t len = 0;
    int ok = 0;
    char *raw = aster_read_file(path, &len, &ok);
    if (!raw) BADF("could not read %s", path);

    const unsigned char *buf = (const unsigned char *)raw;
    if (len < VOCAB_HEAD || memcmp(buf, VOCAB_MAGIC, 8) != 0) {
        free(raw);
        BADF("%s is not an Aster vocabulary file", path);
    }
    uint32_t nm = get_u32le(buf + 8);
    if (nm > ASTER_MAX_MERGES) {
        free(raw);
        BADF("%s claims %u merges, but this build stores at most %d. It was made by a "
             "build with a larger ASTER_MAX_MERGES.", path, nm, ASTER_MAX_MERGES);
    }
    if (len != VOCAB_HEAD + 4u * (size_t)nm) {
        free(raw);
        BADF("%s is %zu bytes but its header describes %u bytes. The file is truncated "
             "or was edited by hand.", path, len, VOCAB_HEAD + 4u * nm);
    }

    AsterVocab *v = (AsterVocab *)aster_xcalloc(1, sizeof *v);
    aster_vocab_init(v);
    v->n_merges = (int)nm;
    v->bytes_per_token = get_f64le(buf + 12);
    for (uint32_t i = 0; i < nm; ++i) {
        const unsigned char *p = buf + VOCAB_HEAD + 4 * (size_t)i;
        v->merge_a[i] = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
        v->merge_b[i] = (uint16_t)((uint16_t)p[2] | ((uint16_t)p[3] << 8));
    }
    free(raw);

    /* Rebuilds the derived tables AND re-validates the merge table, so a file
     * edited by hand into something self-inconsistent is rejected here with a
     * reason rather than misbehaving at encode time. */
    if (aster_vocab_finalize(v, 1, err, errlen) != 0) {
        aster_vocab_free(v);
        free(v);
        return NULL;
    }
    return v;
#undef BADF
}
