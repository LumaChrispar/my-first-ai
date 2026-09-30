/* main.c - command-line dispatch.
 *
 * Commands: train, vocab, generate, serve, eval, selftest, help.
 * Exit codes: 0 success, 1 runtime failure, 2 usage error.
 */
#include "jsonstr.h"
#include "model.h"
#include "server.h"
#include "tokenizer.h"
#include "train.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DATA_FILES 64

static void print_help(void) {
    aster_info("Aster - tiny local language model (checkpoint format %d, arch %d, tokenizer %d)",
               ASTER_CHECKPOINT_FORMAT, ASTER_ARCH_VERSION, ASTER_TOKENIZER_VERSION);
    puts("");
    puts("USAGE");
    puts("  aster <command> [options]");
    puts("");
    puts("COMMANDS");
    puts("  train      build a checkpoint from local text files");
    puts("  vocab      learn a sub-word vocabulary, or print one that was learned");
    puts("  generate   load a checkpoint and print a continuation");
    puts("  serve      run the local chat server and browser interface");
    puts("  eval       report held-out loss for a checkpoint on a dataset");
    puts("  selftest   check the tokenizer, bounds handling, and checkpoint IO");
    puts("  help       show this text");
    puts("");
    puts("TRAIN");
    puts("  --data PATH           UTF-8 text file, repeatable (required)");
    puts("  --validation PATH     held-out text file, repeatable");
    puts("  --out PATH            checkpoint to write (default models/aster-small.bin)");
    puts("  --mode text|chat      text = next-token corpus, chat = JSONL dialogues");
    puts("  --seed N              random seed (default 1234)");
    puts("  --steps N             optimizer steps (default 1000)");
    puts("  --batch N             windows per step (default 8)");
    puts("  --block N             tokens per window, <= context (default 64)");
    puts("  --lr X                learning rate (default 1e-3)");
    puts("  --log-every N         log interval in steps (default 25)");
    puts("  --merges N            BPE merges to learn (default 1024, 0 = byte only)");
    puts("  --vocab PATH          reuse a learned .vocab instead of learning one");
    puts("");
    puts("VOCAB");
    puts("  --data PATH           text to learn from, repeatable (required to learn)");
    puts("  --mode text|chat      must match how the model will be trained");
    puts("  --merges N            how many merges to learn (default 1024)");
    puts("  --out PATH            .vocab file to write");
    puts("  --in PATH             print an existing .vocab file instead of learning");
    puts("");
    puts("  Merges are learned from --data only and NEVER from --validation.");
    puts("  A vocabulary fitted on the held-out set leaks it, which would make");
    puts("  the held-out loss meaningless. There is no flag to do that.");
    puts("");
    puts("GENERATE / SERVE");
    puts("  --model PATH          checkpoint to load");
    puts("  --prompt TEXT         prompt for generate");
    puts("  --system TEXT         text between SYSTEM and USER; must match training");
    puts("  --max-new-tokens N    cap on generated tokens (default 80)");
    puts("  --temp X              sampling temperature; 0 or less means greedy (default 0)");
    puts("  --top-k N             keep only the N most likely tokens (0 = off)");
    puts("  --seed N              sampling seed (default 1234)");
    puts("  --host ADDR           loopback address to bind (default 127.0.0.1)");
    puts("  --port N              port to bind (default 8080)");
    puts("");
    puts("EVAL");
    puts("  --data PATH           held-out file, repeatable");
    puts("  --mode text|chat      must match how the model was trained");
    puts("  --block N             tokens per window (default 64)");
    puts("");
    puts("EXAMPLES (PowerShell)");
    puts("  .\\aster.exe vocab --mode chat --data data/demo_chat.jsonl `");
    puts("      --merges 1024 --out models/aster-small.vocab");
    puts("  .\\aster.exe train --mode chat --steps 600 `");
    puts("      --data data/demo_chat.jsonl --validation data/demo_valid.jsonl `");
    puts("      --vocab models/aster-small.vocab `");
    puts("      --out models/aster-small.bin --seed 1234");
    puts("  .\\aster.exe generate --model models/aster-small.bin --prompt \"Who are you?\"");
    puts("  .\\aster.exe serve --model models/aster-small.bin --port 8080");
    puts("");
    puts("  --vocab is optional: train learns a vocabulary from --data when you");
    puts("  do not pass one. Pass it explicitly to reuse the same tokenizer across");
    puts("  runs, which is what makes two checkpoints comparable.");
    puts("");
    puts("  If the corpus has non-empty system turns, pass the same text here,");
    puts("  e.g. --system \"You are Aster.\" A mismatch makes replies meaningless.");
}

/* ------------------------------------------------------------ arg parsing */

typedef struct {
    const char *data[MAX_DATA_FILES];
    int         n_data;
    const char *valid[MAX_DATA_FILES];
    int         n_valid;
    const char *out;
    const char *model;
    const char *vocab;      /* --vocab, or --in for the vocab verb */
    const char *prompt;
    const char *system;
    const char *host;
    const char *mode;
    int    port;
    int    steps, batch, block, log_every, max_new, top_k, has_port, has_max_new;
    int    merges, has_merges;
    float  lr, temp;
    int    has_temp, has_lr;
    uint32_t seed;
    int    has_seed;
} Args;

static int arg_int(const char *s, const char *what) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!end || *end != '\0') aster_fail("%s expects a whole number, got \"%s\"", what, s);
    if (v < -1000000L || v > 1000000L) aster_fail("%s value %s is out of range", what, s);
    return (int)v;
}

static float arg_float(const char *s, const char *what) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (!end || *end != '\0') aster_fail("%s expects a number, got \"%s\"", what, s);
    if (!isfinite(v)) aster_fail("%s must be finite", what);
    return (float)v;
}

static void parse_args(int argc, char **argv, Args *a, int start) {
    memset(a, 0, sizeof *a);
    a->out = "models/aster-small.bin";
    a->host = "127.0.0.1";
    a->port = 8080;
    a->mode = "text";
    a->block = 64;
    a->steps = -1;      /* -1 means "leave the default" */
    a->log_every = -1;
    a->seed = 0;
    a->has_seed = 0;
    a->temp = 0.0f;
    a->has_temp = 0;
    a->merges = ASTER_DEFAULT_MERGES;
    a->has_merges = 0;

#define NEXT() do { if (i + 1 >= argc) aster_fail("%s needs a value", argv[i]); } while (0)
    for (int i = start; i < argc; ++i) {
        const char *k = argv[i];
        if      (!strcmp(k, "--data"))      { NEXT(); if (a->n_data >= MAX_DATA_FILES) aster_fail("too many --data files"); a->data[a->n_data++] = argv[++i]; }
        else if (!strcmp(k, "--validation")){ NEXT(); if (a->n_valid >= MAX_DATA_FILES) aster_fail("too many --validation files"); a->valid[a->n_valid++] = argv[++i]; }
        else if (!strcmp(k, "--out"))       { NEXT(); a->out = argv[++i]; }
        else if (!strcmp(k, "--model"))     { NEXT(); a->model = argv[++i]; }
        else if (!strcmp(k, "--vocab"))     { NEXT(); a->vocab = argv[++i]; }
        else if (!strcmp(k, "--in"))        { NEXT(); a->vocab = argv[++i]; }
        else if (!strcmp(k, "--merges"))    { NEXT(); a->merges = arg_int(argv[++i], "--merges"); a->has_merges = 1; }
        else if (!strcmp(k, "--prompt"))    { NEXT(); a->prompt = argv[++i]; }
        else if (!strcmp(k, "--system"))    { NEXT(); a->system = argv[++i]; }
        else if (!strcmp(k, "--host"))      { NEXT(); a->host = argv[++i]; }
        else if (!strcmp(k, "--mode"))      { NEXT(); a->mode = argv[++i]; }
        else if (!strcmp(k, "--seed"))      { NEXT(); a->seed = (uint32_t)arg_int(argv[++i], "--seed"); a->has_seed = 1; }
        else if (!strcmp(k, "--steps"))     { NEXT(); a->steps = arg_int(argv[++i], "--steps"); }
        else if (!strcmp(k, "--batch"))     { NEXT(); a->batch = arg_int(argv[++i], "--batch"); }
        else if (!strcmp(k, "--block"))     { NEXT(); a->block = arg_int(argv[++i], "--block"); }
        else if (!strcmp(k, "--log-every")) { NEXT(); a->log_every = arg_int(argv[++i], "--log-every"); }
        else if (!strcmp(k, "--lr"))        { NEXT(); a->lr = arg_float(argv[++i], "--lr"); a->has_lr = 1; }
        else if (!strcmp(k, "--max-new-tokens")) { NEXT(); a->max_new = arg_int(argv[++i], "--max-new-tokens"); a->has_max_new = 1; }
        else if (!strcmp(k, "--temp"))      { NEXT(); a->temp = arg_float(argv[++i], "--temp"); a->has_temp = 1; }
        else if (!strcmp(k, "--top-k"))     { NEXT(); a->top_k = arg_int(argv[++i], "--top-k"); }
        else if (!strcmp(k, "--port"))      { NEXT(); a->port = arg_int(argv[++i], "--port"); a->has_port = 1; }
        else if (!strcmp(k, "--help") || !strcmp(k, "-h")) { print_help(); exit(0); }
        else aster_fail("unknown option \"%s\". Run \"aster help\" for the list.", k);
    }
#undef NEXT
    if (!strcmp(a->mode, "chat")) { }
    else if (!strcmp(a->mode, "text")) { }
    else aster_fail("--mode must be \"text\" or \"chat\" (got \"%s\")", a->mode);
    if (a->merges < 0 || a->merges > ASTER_MAX_MERGES)
        aster_fail("--merges must be between 0 and %d (got %d)", ASTER_MAX_MERGES, a->merges);
}

/* --------------------------------------------------------- vocabulary -----
 * The learner sees text through dataset_scan_text, never a file path, so it is
 * structurally impossible to point it at the validation split. That is
 * deliberate: a merge table fitted on held-out data leaks it, and every number
 * computed afterwards stops meaning what it appears to mean. */

static const char *file_name_of(const char *path);

typedef struct { AsterVocabLearner *l; size_t bytes; size_t spans; } VocabSink;

static void vocab_sink_text(void *ud, const char *s, size_t n) {
    VocabSink *vs = (VocabSink *)ud;
    bpe_learner_add_text(vs->l, s, n);
    vs->bytes += n;
    vs->spans++;
}

/* Loads a.vocab if it names one, otherwise learns a fresh vocabulary from the
 * training split. Returns a heap vocabulary the caller frees with
 * aster_vocab_free + free, or NULL with a reason in err. */
static AsterVocab *vocab_resolve(Args *a, const DataSpec *ts, char *err, size_t errlen) {
    if (a->vocab) {
        AsterVocab *v = aster_vocab_load(a->vocab, err, errlen);
        if (!v) {
            if (err && errlen) {
                char tmp[512];
                snprintf(tmp, sizeof tmp, "%s. The checkpoint carries its own copy of the "
                         "merge table, so --vocab is only for reusing a tokenizer between "
                         "runs, never for changing how a trained model reads text.", err);
                snprintf(err, errlen, "%s", tmp);
            }
            return NULL;
        }
        aster_info("vocabulary: %s (%d merges, %d tokens, %.2f bytes/token)",
                   file_name_of(a->vocab), v->n_merges, aster_vocab_size(v), v->bytes_per_token);
        return v;
    }

    AsterVocabLearner *L = bpe_learner_new(a->merges);
    if (!L) {
        snprintf(err, errlen, "could not allocate a vocabulary learner");
        return NULL;
    }
    VocabSink sink = { L, 0, 0 };
    if (dataset_scan_text(ts, vocab_sink_text, &sink, err, errlen) != 0) {
        bpe_learner_free(L);
        return NULL;
    }
    AsterVocab *v = (AsterVocab *)aster_xcalloc(1, sizeof *v);
    if (bpe_learner_finish(L, v, err, errlen) != 0) {
        aster_vocab_free(v);
        free(v);
        bpe_learner_free(L);
        return NULL;
    }
    /* Read the statistics BEFORE freeing the learner; they live in it. */
    size_t words = 0, distinct = 0;
    int merges = 0;
    bpe_learner_stats(L, &words, &distinct, &merges);
    bpe_learner_free(L);

    aster_info("vocabulary: learned from %zu span(s), %zu bytes of training text",
               sink.spans, sink.bytes);
    aster_info("  %zu words seen, %zu distinct kept, %d merges, %d tokens, %.2f bytes/token%s",
               words, distinct, v->n_merges, aster_vocab_size(v), v->bytes_per_token,
               v->n_merges ? "" : " (byte fallback: no pair was frequent enough to merge)");
    if (a->merges > 0 && v->n_merges < a->merges)
        aster_warn("asked for %d merges but the corpus only supported %d. A small corpus "
                   "runs out of frequent pairs; this is not an error.", a->merges, v->n_merges);
    return v;
}

/* Prints a vocabulary. This is a real command, not debug output: it is how a
 * user checks that the merges are the ones they expect, and how a surprising
 * one gets spotted. */
static void vocab_print(AsterVocab *v) {
    aster_info("vocabulary %s: %d merges, %d tokens, %.3f bytes/token",
               "file", v->n_merges, aster_vocab_size(v), v->bytes_per_token);
    puts("");
    puts("  id   bytes   spelling");
    puts("  ---- ------- ------------------------------------");
    for (int id = 0; id < aster_vocab_size(v); ++id) {
        if (tok_is_structural((uint16_t)id)) continue;
        aster_info("  %4d %7d  %s", id, (int)v->tok_len[id], tok_name(v, (uint16_t)id));
    }
    puts("");
    aster_info("The first 256 ids are raw bytes and are not listed; they always exist, which");
    aster_info("is what lets the encoder represent text the merges never saw.");
}

static int cmd_vocab(int argc, char **argv) {
    Args a;
    parse_args(argc, argv, &a, 2);

    char err[512];
    if (a.n_data < 1 && !a.vocab) {
        fprintf(stderr, "error: give --data to learn a vocabulary, or --in to print one\n");
        return 2;
    }
    puts("");
    if (a.vocab) {
        AsterVocab *v = aster_vocab_load(a.vocab, err, sizeof err);
        if (!v) { fprintf(stderr, "error: %s\n", err); return 1; }
        vocab_print(v);
        aster_vocab_free(v);
        free(v);
        return 0;
    }

    DataSpec ts;
    memset(&ts, 0, sizeof ts);
    ts.paths = a.data;
    ts.n_paths = a.n_data;
    ts.block = a.block;
    ts.mode = (strcmp(a.mode, "chat") == 0) ? ASTER_DATA_CHAT : ASTER_DATA_TEXT;

    AsterVocab *v = vocab_resolve(&a, &ts, err, sizeof err);
    if (!v) { fprintf(stderr, "error: could not build a vocabulary: %s\n", err); return 1; }

    int rc = 0;
    if (a.out) {
        if (aster_vocab_save(v, a.out, err, sizeof err) != 0) {
            fprintf(stderr, "error: %s\n", err);
            rc = 1;
        } else {
            aster_info("wrote %s", a.out);
        }
    } else {
        vocab_print(v);
    }
    aster_vocab_free(v);
    free(v);
    return rc;
}

static const char *file_name_of(const char *path) {
    const char *a = strrchr(path, '/'), *b = strrchr(path, '\\');
    const char *s = (a && b) ? (a > b ? a : b) : (a ? a : b);
    return s ? s + 1 : path;
}

/* -------------------------------------------------------------- selftest */

static int check(int ok, const char *what, int *failures) {
    aster_info("%-58s %s", what, ok ? "pass" : "FAIL");
    if (!ok) (*failures)++;
    return ok;
}

/* Loss of a tiny fixed window, used by the gradient check below. */
static double gc_loss(AsterModel *m, AsterActs *a, const uint16_t *x,
                      const uint16_t *y, const float *w) {
    double l = 0.0;
    aster_forward(m, a, x, y, w, &l);
    return l;
}

/* Central difference of the loss with respect to params[i], Richardson
 * extrapolated from step h and h/2 so the O(h^2) truncation term cancels
 * without needing a smaller (and noisier) step.
 *
 * `cur` must be computed before the extrapolation. Folding the new difference
 * into `d` and then reusing `d` on the next step silently evaluates
 * (4*D(h) - D(h))/3 == D(h): a plain central difference, with the h/2 passes
 * computed and discarded. That mistake hides the O(h^2) error instead of
 * cancelling it -- at h=1e-2 it is worth about 20% on the larger gradients,
 * which is far enough to make a correct backward pass look broken. */
static double gc_numeric(AsterModel *m, AsterActs *a, const uint16_t *x,
                         const uint16_t *y, const float *w, int i) {
    const float o = m->params[i];
    double h = 1e-2, d = 0.0, prev = 0.0;
    for (int k = 0; k < 2; ++k) {
        m->params[i] = (float)(o + h);
        double p = gc_loss(m, a, x, y, w);
        m->params[i] = (float)(o - h);
        double q = gc_loss(m, a, x, y, w);
        double cur = (p - q) / (2.0 * h);
        d = (k == 0) ? cur : (4.0 * cur - prev) / 3.0;
        prev = cur;
        h *= 0.5;
    }
    m->params[i] = o;
    return d;
}

/* A small learned vocabulary, built from a fixed corpus with a fixed merge
 * count so it is identical on every run. The corpus is repeated because a pair
 * must be seen at least twice to be worth merging -- a vocabulary learned from
 * a string with no repetition learns nothing, which is correct behaviour and
 * makes a bad test. */
static const char *ST_CORPUS =
    "the model is honest and the model is small. "
    "the model is honest and the model is small. "
    "the model is honest and the model is small. "
    "the model is honest and the model is small. "
    "the model is honest and the model is small. "
    "the model is honest and the model is small. "
    "the model is honest and the model is small. "
    "the model is honest and the model is small. "
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog. "
    "a tiny language model can learn to say the is and the model. "
    "a tiny language model can learn to say the is and the model. "
    "a tiny language model can learn to say the is and the model. ";

static AsterVocabLearner *st_learner(int merges, const char *text) {
    AsterVocabLearner *L = bpe_learner_new(merges);
    if (!L) aster_fail("selftest: could not allocate a vocabulary learner");
    bpe_learner_add_text(L, text, strlen(text));
    return L;
}

/* Fills *v with a learned vocabulary. Fails the whole run loudly rather than
 * returning a silently different one: a self-test that quietly tested a
 * different tokenizer than the one it names is worse than no test. */
static void st_vocab(AsterVocab *v, int merges, const char *text) {
    char err[256];
    aster_vocab_init(v);
    AsterVocabLearner *L = st_learner(merges, text);
    if (bpe_learner_finish(L, v, err, sizeof err) != 0) {
        bpe_learner_free(L);
        aster_fail("selftest: could not learn a test vocabulary: %s", err);
    }
    bpe_learner_free(L);
}

static void st_vocab_free(AsterVocab *v) { aster_vocab_free(v); }

/* encode -> decode -> encode, compared id by id. This is the property the
 * whole tokenizer rests on, and it is checked over inputs chosen to break it
 * rather than over one friendly sentence. */
static int st_roundtrip(const AsterVocab *v, const char *s, size_t n) {
    TokenList a, b;
    token_list_init(&a);
    token_list_init(&b);
    if (tok_encode(v, s, n, &a) != 0) { token_list_free(&a); return 0; }
    char *back = (char *)aster_xmalloc(a.count * ASTER_BPE_MAX_TOK_BYTES + 16);
    size_t bn = tok_decode(v, a.ids, a.count, back, a.count * ASTER_BPE_MAX_TOK_BYTES + 16, 0);
    int ok = (bn == n) && (n == 0 || memcmp(back, s, n) == 0);
    if (ok) ok = (tok_encode(v, back, bn, &b) == 0) && a.count == b.count &&
                 (a.count == 0 || memcmp(a.ids, b.ids, a.count * sizeof(uint16_t)) == 0);
    free(back);
    token_list_free(&a);
    token_list_free(&b);
    return ok;
}

static int cmd_selftest(void) {
    int failures = 0;
    aster_info("Aster self-test");
    puts("");

    /* The vocabulary every test below uses, plus a zero-merge one. The
     * zero-merge case is not a throwaway: a vocabulary with no merges is the
     * old byte tokenizer, so it is the regression baseline every new claim is
     * measured against. */
    AsterVocab V, V0;
    st_vocab(&V, 64, ST_CORPUS);
    aster_vocab_init(&V0);
    char verr[256];
    if (aster_vocab_finalize(&V0, 1, verr, sizeof verr) != 0)
        aster_fail("selftest: the byte-fallback vocabulary is invalid: %s", verr);
    aster_info("test vocabulary: %d tokens, %d merges, %d max token bytes",
               aster_vocab_size(&V), V.n_merges, aster_vocab_max_token_bytes(&V));
    puts("");

    /* ---- tokenizer round trips ---- */
    {
        const char *ascii = "The quick brown fox jumps over the lazy dog. 0123456789";
        TokenList tl;
        token_list_init(&tl);
        tok_encode(&V, ascii, strlen(ascii), &tl);
        char back[256];
        tok_decode(&V, tl.ids, tl.count, back, sizeof back, 0);
        check(strcmp(ascii, back) == 0, "ASCII encode/decode round trip", &failures);
        token_list_free(&tl);
    }
    {
        /* Valid multi-byte UTF-8, written out explicitly so the source file
         * encoding cannot affect the test. */
        const char utf8[] = "caf\xC3\xA9 na\xC3\xAFve \xE2\x9C\x93 \xF0\x9F\x9A\x80 done";
        TokenList tl;
        token_list_init(&tl);
        tok_encode(&V, utf8, sizeof utf8 - 1, &tl);
        char back[256];
        tok_decode(&V, tl.ids, tl.count, back, sizeof back, 0);
        int ok = (strcmp(utf8, back) == 0) && utf8_is_valid(back, strlen(back));
        check(ok, "UTF-8 encode/decode round trip", &failures);
        token_list_free(&tl);
    }
    {
        /* Special markers must never collide with byte tokens, and the byte
         * range must stay below them however many merges are learned. */
        int ok = TOK_SYSTEM > 255 && TOK_USER > 255 && TOK_ASSISTANT > 255 &&
                 TOK_END > 255 && TOK_PAD == 260 &&
                 ASTER_BASE_VOCAB == 261 && ASTER_MAX_VOCAB == 1285 &&
                 aster_vocab_size(&V0) == 261 && aster_vocab_size(&V) == 261 + V.n_merges;
        check(ok, "special markers are distinct and above the byte range", &failures);
    }
    {
        /* A long input must be reported, not written past the buffer. The old
         * version of this test asserted "8191 bytes give 8191 tokens", which is
         * exactly what a merge table stops being true for -- a run of identical
         * bytes is what BPE compresses hardest. It is now stated as the property
         * that still has to hold: the decode is bounded by the buffer, and the
         * text is a prefix of the input. */
        char big[8192];
        memset(big, 'x', sizeof big - 1);
        big[sizeof big - 1] = '\0';
        TokenList tl;
        token_list_init(&tl);
        tok_encode(&V, big, sizeof big - 1, &tl);
        char small[16];
        size_t n = tok_decode(&V, tl.ids, tl.count, small, sizeof small, 0);
        int ok = (n == sizeof small - 1) && small[sizeof small - 1] == '\0' &&
                 (tl.count == 0 || memcmp(small, big, n) == 0) &&
                 tl.count <= sizeof big - 1;
        check(ok, "decode respects a tiny output buffer", &failures);
        token_list_free(&tl);
    }
    {
        /* Malformed UTF-8 must be replaced, never crash or pass through. */
        const char bad[] = "ok \xC3 bad \xFF\xFE end";
        char clean[64];
        int replaced = utf8_sanitize(bad, sizeof bad - 1, clean, sizeof clean);
        int ok = replaced && utf8_is_valid(clean, strlen(clean)) && strstr(clean, "ok ") != NULL;
        check(ok, "invalid UTF-8 is replaced, not passed through", &failures);
    }

    /* ---- the round-trip property, over inputs chosen to break it ---- */
    {
        int ok = 1;
        /* Empty. */
        ok &= st_roundtrip(&V, "", 0);
        /* A single space, a single byte, a single control byte. */
        ok &= st_roundtrip(&V, " ", 1);
        ok &= st_roundtrip(&V, "x", 1);
        ok &= st_roundtrip(&V, "\x7F", 1);
        /* Every byte value except NUL -- all 255 of them must survive. NUL is
         * excluded because it is a documented ENCODING ERROR rather than
         * something that round-trips: skipping it would silently break the
         * property for any string containing one, and a string can contain
         * one. The next check asserts that it is reported, not swallowed. */
        {
            char all[255];
            for (int i = 0; i < 255; ++i) all[i] = (char)(i + 1);
            ok &= st_roundtrip(&V, all, 255);
        }
        /* Lone UTF-8 continuation bytes: invalid on their own, and exactly the
         * case where a tokenizer is tempted to drop bytes to stay "valid". */
        ok &= st_roundtrip(&V, "\x80", 1);
        ok &= st_roundtrip(&V, "\xBF\xBF", 2);
        /* 4-byte emoji, and a truncated one. */
        ok &= st_roundtrip(&V, "\xF0\x9F\x9A\x80", 4);
        ok &= st_roundtrip(&V, "\xF0\x9F", 2);
        /* A script the merges never saw, and one where a merge exists so the
         * interesting case is a mix of merged and unmerged text. */
        ok &= st_roundtrip(&V, "\xE4\xBD\xA0\xE5\xA5\xBD", 6);
        ok &= st_roundtrip(&V, "the model \xE4\xBD\xA0\xE5\xA5\xBD is", 19);
        /* A long run of one repeated byte: the input BPE compresses hardest,
         * and the input a length cap is most likely to trip on. */
        {
            char run[512];
            memset(run, 'a', sizeof run);
            ok &= st_roundtrip(&V, run, sizeof run);
        }
        /* Pseudo-random bytes from a fixed seed, so a failure is reproducible.
         * Byte 0 is mapped to 1: this generator does produce NULs, and a NUL is
         * an error by design, which is asserted separately above. */
        {
            unsigned char r[1024];
            uint32_t st = 12345u;
            for (size_t i = 0; i < sizeof r; ++i) {
                unsigned b = (unsigned)(aster_rng_u32(&st) & 0xFF);
                r[i] = (unsigned char)(b ? b : 1);
            }
            ok &= st_roundtrip(&V, (const char *)r, sizeof r);
        }
        check(ok, "encode(decode(encode(s))) == s over adversarial inputs", &failures);
    }
    {
        /* An embedded NUL is an error, not something to skip: skipping it would
         * quietly break the round-trip property for any string containing one,
         * and a string can contain one. */
        TokenList tl;
        token_list_init(&tl);
        int rc = tok_encode(&V, "a\0b", 3, &tl);
        check(rc == -2, "an embedded NUL is reported as an encoding error", &failures);
        token_list_free(&tl);
    }
    {
        /* Byte fallback: text the merges never saw must still round-trip, and
         * must do so through byte ids rather than by failing. */
        const char *rare = "\x01\x02\xFE\xFD zqxjkvbnm \x10\x11\x12";
        TokenList tl;
        token_list_init(&tl);
        int rc = tok_encode(&V, rare, strlen(rare), &tl);
        int all_bytes = 1;
        for (size_t i = 0; i < tl.count; ++i) if (tl.ids[i] > ASTER_BYTE_MAX) all_bytes = 0;
        char back[128];
        size_t bn = tok_decode(&V, tl.ids, tl.count, back, sizeof back, 0);
        check(rc == 0 && all_bytes && bn == strlen(rare) && memcmp(back, rare, bn) == 0,
              "unseen text falls back to byte tokens and still round-trips", &failures);
        token_list_free(&tl);
    }
    {
        /* Determinism: same input, same ids, twice. A tokenizer that depends on
         * hash iteration order produces a model that cannot be reproduced, and
         * the only symptom is a loss number that moves between identical runs. */
        const char *s = "the model is honest, the model is small";
        TokenList a, b;
        token_list_init(&a);
        token_list_init(&b);
        tok_encode(&V, s, strlen(s), &a);
        tok_encode(&V, s, strlen(s), &b);
        check(a.count == b.count &&
              memcmp(a.ids, b.ids, a.count * sizeof(uint16_t)) == 0,
              "encoding the same text twice gives identical ids", &failures);
        token_list_free(&a);
        token_list_free(&b);
    }
    {
        /* Determinism of LEARNING: same corpus, same merge table. */
        AsterVocab A, B;
        st_vocab(&A, 64, ST_CORPUS);
        st_vocab(&B, 64, ST_CORPUS);
        int ok = (A.n_merges == B.n_merges);
        for (int i = 0; ok && i < A.n_merges; ++i)
            if (A.merge_a[i] != B.merge_a[i] || A.merge_b[i] != B.merge_b[i]) ok = 0;
        check(ok, "learning the same corpus twice gives an identical merge table", &failures);
        st_vocab_free(&A);
        st_vocab_free(&B);
    }
    {
        /* Merges must actually be learned and must actually be used, or all of
         * the above passes on a tokenizer that has quietly done nothing. */
        const char *s = "the model is honest";
        TokenList tl;
        token_list_init(&tl);
        tok_encode(&V, s, strlen(s), &tl);
        int merged = 0;
        for (size_t i = 0; i < tl.count; ++i) if (tl.ids[i] >= TOK_FIRST_MERGE) merged = 1;
        check(V.n_merges > 0 && merged && tl.count < strlen(s),
              "learned merges are applied, and the vocabulary is not empty", &failures);
        if (merged)
            aster_info("  \"%s\" -> %zu tokens for %zu bytes, first is \"%s\"",
                       s, tl.count, strlen(s), tok_name(&V, tl.ids[0]));
        token_list_free(&tl);
    }
    {
        /* A vocabulary file round-trips, and the reloaded one encodes
         * identically. */
        char err[256];
        check(aster_vocab_save(&V, "models/selftest.vocab", err, sizeof err) == 0,
              "a vocabulary file is written", &failures);
        AsterVocab *R = aster_vocab_load("models/selftest.vocab", err, sizeof err);
        int ok = (R != NULL) && R->n_merges == V.n_merges;
        if (ok) {
            const char *s = "the model is honest and small";
            TokenList a, b;
            token_list_init(&a);
            token_list_init(&b);
            tok_encode(&V, s, strlen(s), &a);
            tok_encode(R, s, strlen(s), &b);
            ok = a.count == b.count &&
                 memcmp(a.ids, b.ids, a.count * sizeof(uint16_t)) == 0;
            token_list_free(&a);
            token_list_free(&b);
        }
        check(ok, "a reloaded vocabulary file encodes identically", &failures);
        if (R) { aster_vocab_free(R); free(R); }
        /* A file that is not a vocabulary, and one whose length disagrees with
         * its own header. */
        FILE *f = fopen("models/selftest-badvocab.bin", "wb");
        if (f) { fwrite("not a vocabulary, just some text here", 1, 34, f); fclose(f); }
        check(aster_vocab_load("models/selftest-badvocab.bin", err, sizeof err) == NULL,
              "a non-vocabulary file is rejected", &failures);
        f = fopen("models/selftest-vocab-trunc.bin", "wb");
        if (f) { fwrite("ASTERVB1", 1, 8, f); fwrite("\x40\x00\x00\x00", 1, 4, f); fclose(f); }
        check(aster_vocab_load("models/selftest-vocab-trunc.bin", err, sizeof err) == NULL,
              "a truncated vocabulary file is rejected", &failures);
        remove("models/selftest.vocab");
        remove("models/selftest-badvocab.bin");
        remove("models/selftest-vocab-trunc.bin");
    }

    /* ---- parameter count matches the documented architecture ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg, &V);
        int n = aster_param_count(&cfg);
        aster_info("default architecture parameter count: %d (%.4f million)", n, n / 1e6);
        /* tok_emb + pos_emb + 2 layers + final norm, from the layout:
         * V*64 + 128*64 + 2*(2*64 + 4*(64*64) + 2*64 + 64*256+256 + 256*64) + 2*64 */
        int expect = cfg.vocab * 64 + 128 * 64 +
                     2 * (2 * 64 + 4 * (64 * 64) + 2 * 64 + 64 * 256 + 256 + 256 * 64) +
                     2 * 64;
        check(n == expect, "parameter count matches the documented layout", &failures);
        /* Weight tying means every extra token costs exactly d_model and
         * nothing else changes: the whole body is untouched. The build spec's
         * headline figure rests on this, so it is asserted rather than left to
         * the arithmetic above. */
        AsterConfig cfg0;
        aster_config_default(&cfg0, &V0);
        int n0 = aster_param_count(&cfg0);
        check(n - n0 == V.n_merges * cfg.d_model,
              "each added token costs exactly d_model parameters", &failures);
        aster_info("  %d tokens vs %d byte-only: +%d parameters, body unchanged at %d",
                   cfg.vocab, cfg0.vocab, n - n0, n - (cfg.vocab - cfg0.vocab) * cfg.d_model);
    }

    /* ---- forward pass produces finite values and a loss near ln(vocab) ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg, &V);
        AsterModel *m = aster_model_new(&cfg, &V, 7u, 1);
        const int B = 2, T = 16;
        AsterActs *a = aster_acts_new(B, T, &cfg);
        uint16_t *x = (uint16_t *)aster_xmalloc((size_t)B * T * sizeof(uint16_t));
        uint16_t *y = (uint16_t *)aster_xmalloc((size_t)B * T * sizeof(uint16_t));
        float *w = (float *)aster_xmalloc((size_t)B * T * sizeof(float));
        for (int i = 0; i < B * T; ++i) { x[i] = (uint16_t)(i * 7 % 256); y[i] = (uint16_t)((i * 13) % 256); w[i] = 1.0f; }
        double loss = aster_forward(m, a, x, y, w, NULL);
        double expect0 = log((double)cfg.vocab);
        int ok = isfinite(loss) && fabs(loss - expect0) < 0.5;
        check(ok, "untrained forward pass is finite and near ln(vocab)", &failures);
        aster_info("  untrained loss %.4f, ln(%d) = %.4f", loss, cfg.vocab, expect0);

        /* Every gradient must be finite after a backward pass. */
        aster_backward(m, a, x, y, w);
        int nonfinite = 0;
        for (int i = 0; i < m->off.total; ++i) if (!isfinite(m->grads[i])) { nonfinite++; break; }
        check(nonfinite == 0, "every gradient is finite after a backward pass", &failures);

        /* Padding must not move the loss. */
        float *w0 = (float *)aster_xcalloc((size_t)B * T, sizeof(float));
        for (int i = 0; i < T; ++i) { w[i] = 1.0f; w0[i] = 0.0f; }
        double l1 = aster_forward(m, a, x, y, w, NULL);
        double l0 = aster_forward(m, a, x, y, w0, NULL);
        check(l0 == 0.0 && isfinite(l1), "a fully masked batch contributes no loss", &failures);

        /* A fully masked batch must also produce bitwise-zero gradients. A
         * masked position that still leaks gradient trains the model on text it
         * was supposed only to read, and the loss curve still looks healthy --
         * so this is asserted bitwise rather than with a tolerance, where a
         * small leak would hide.
         *
         * The buffer is zeroed first because aster_backward ACCUMULATES: with a
         * total weight of 0 it returns before writing anything, which is correct
         * but means a pre-dirtied buffer would still be dirty on return. */
        memset(m->grads, 0, sizeof(float) * (size_t)m->off.total);
        aster_forward(m, a, x, y, w0, NULL);
        aster_backward(m, a, x, y, w0);
        int nonzero = 0;
        for (int i = 0; i < m->off.total; ++i) if (m->grads[i] != 0.0f) { nonzero++; break; }
        check(nonzero == 0, "a fully masked batch produces bitwise-zero gradients", &failures);

        free(x); free(y); free(w); free(w0);
        aster_acts_free(a);
        aster_model_free(m);
    }

    /* ---- out-of-range token ids are rejected, not read out of bounds ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg, &V);
        AsterModel *m = aster_model_new(&cfg, &V, 7u, 0);
        AsterActs *a = aster_acts_new(1, 8, &cfg);
        uint16_t bad[8];
        for (int i = 0; i < 8; ++i) bad[i] = (uint16_t)cfg.vocab;   /* one past the end */
        double r = aster_forward(m, a, bad, NULL, NULL, NULL);
        check(r == -1.0, "an out-of-range token id is rejected", &failures);
        aster_acts_free(a);
        aster_model_free(m);
    }
    {
        /* A vocab size outside the range a vocabulary can produce must be
         * refused by validation, with a reason rather than a crash. The
         * matching check in aster_model_new is a hard exit, so it is exercised
         * by the train path rather than from here. */
        AsterConfig cfg;
        char err[160];
        int ok = 1;
        aster_config_default(&cfg, &V);
        cfg.vocab = ASTER_BASE_VOCAB - 1;
        if (aster_config_validate(&cfg, err, sizeof err) == 0) ok = 0;
        aster_config_default(&cfg, &V);
        cfg.vocab = ASTER_MAX_VOCAB + 1;
        if (aster_config_validate(&cfg, err, sizeof err) == 0) ok = 0;
        check(ok, "a vocabulary size outside 261..1285 is refused", &failures);
        /* And the in-range sizes are accepted, so the check above is not just
         * rejecting everything. */
        int good = 1;
        aster_config_default(&cfg, &V);
        if (aster_config_validate(&cfg, err, sizeof err) != 0) good = 0;
        aster_config_default(&cfg, &V0);
        if (aster_config_validate(&cfg, err, sizeof err) != 0) good = 0;
        check(good, "a vocabulary size inside 261..1285 is accepted", &failures);
    }

    /* ---- checkpoint round trip, plus rejection of corrupt files ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg, &V);
        AsterModel *m = aster_model_new(&cfg, &V, 99u, 0);
        for (int i = 0; i < m->off.total; ++i) m->params[i] = (float)i * 1e-4f;
        aster_checkpoint_save(m, "models/selftest.bin", NULL, NULL, 42u, 99u, 80u, 0.0f);

        char err[256];
        uint32_t step = 0, seed = 0;
        AsterModel *r = aster_checkpoint_load("models/selftest.bin", NULL, NULL, &step, &seed, err, sizeof err);
        int ok = (r != NULL) && step == 42 && seed == 99 && r->off.total == m->off.total;
        for (int i = 0; ok && i < m->off.total; ++i) if (r->params[i] != m->params[i]) { ok = 0; break; }
        check(ok, "checkpoint save/load round trip is exact", &failures);

        /* The merge table must travel with the weights. Without this a
         * checkpoint is unusable by a build that does not happen to have the
         * same .vocab file, and the symptom is a model that reads text
         * differently from the one that trained it. */
        int mok = (r != NULL) && r->vocab.n_merges == V.n_merges;
        if (mok) {
            const char *s = "the model is honest";
            TokenList a, b;
            token_list_init(&a);
            token_list_init(&b);
            tok_encode(&m->vocab, s, strlen(s), &a);
            tok_encode(&r->vocab, s, strlen(s), &b);
            mok = a.count == b.count && memcmp(a.ids, b.ids, a.count * sizeof(uint16_t)) == 0;
            token_list_free(&a);
            token_list_free(&b);
        }
        check(mok, "the merge table survives the checkpoint round trip", &failures);
        /* bytes_per_token must survive as a VALUE. An earlier version wrote it
         * with a numeric cast to uint64, which truncates 3.6839 to 3 and reads
         * back as a perfectly plausible 3.0 -- so the bug was invisible in the
         * file and only showed up as a subtly wrong character counter. The
         * value is checked exactly, not approximately, for that reason. */
        check(r != NULL && r->vocab.bytes_per_token == V.bytes_per_token,
              "bytes_per_token survives the checkpoint as a value, not a cast", &failures);
        aster_model_free(r);

        /* A single flipped byte in the parameter block must be caught by the
         * integrity block. This is what README.md claimed a CRC and a SHA-256
         * were for before they were actually computed. */
        {
            size_t flen = 0;
            int fok = 0;
            char *fbuf = aster_read_file("models/selftest.bin", &flen, &fok);
            int caught = 0;
            if (fbuf && flen > 36) {
                for (int off = 0; off < 3 && !caught; ++off) {
                    char path[64];
                    size_t at = flen / 3 + (size_t)off;
                    if (at >= flen - 36) continue;
                    fbuf[at] = (char)(fbuf[at] ^ 0x01);
                    snprintf(path, sizeof path, "models/selftest-flip%d.bin", off);
                    aster_write_file_atomic(path, fbuf, flen);
                    fbuf[at] = (char)(fbuf[at] ^ 0x01);   /* restore for the next try */
                    AsterModel *bad = aster_checkpoint_load(path, NULL, NULL, NULL, NULL, err, sizeof err);
                    if (bad == NULL) caught = 1;
                    else aster_model_free(bad);
                    if (caught) aster_info("  a flipped byte at offset %zu was rejected: %s", at, err);
                    remove(path);
                }
            }
            free(fbuf);
            check(caught, "a single flipped byte is caught by the integrity block", &failures);
        }

        /* Truncated file. */
        FILE *f = fopen("models/selftest.bin", "rb");
        size_t sz = 0;
        if (f) { fseek(f, 0, SEEK_END); sz = (size_t)ftell(f); fclose(f); }
        size_t cut = sz > 40 ? sz / 2 : sz;
        if (sz) {
            char *buf = (char *)aster_xmalloc(cut);
            FILE *rd = fopen("models/selftest.bin", "rb");
            if (rd && fread(buf, 1, cut, rd) == cut) {
                FILE *g = fopen("models/selftest-trunc.bin", "wb");
                if (g) { fwrite(buf, 1, cut, g); fclose(g); }
            }
            /* Both handles must be closed here. Windows refuses to remove a
             * file that still has an open handle, so a leak here silently
             * leaves selftest-trunc.bin behind after a passing run. */
            if (rd) fclose(rd);
            free(buf);
        }
        r = aster_checkpoint_load("models/selftest-trunc.bin", NULL, NULL, NULL, NULL, err, sizeof err);
        check(r == NULL, "a truncated checkpoint is rejected", &failures);
        if (r) aster_model_free(r);
        if (r == NULL) aster_info("  reason: %s", err);

        /* Wrong magic bytes. */
        f = fopen("models/selftest-bad.bin", "wb");
        if (f) { fwrite("not a model at all, just some text", 1, 33, f); fclose(f); }
        r = aster_checkpoint_load("models/selftest-bad.bin", NULL, NULL, NULL, NULL, err, sizeof err);
        check(r == NULL, "a non-checkpoint file is rejected", &failures);
        if (r) aster_model_free(r);

        /* A byte-tokenizer checkpoint must be refused with a message that says
         * what to do. Silently reshaping it would be worse than refusing.
         *
         * The fixture is a real, integrity-valid checkpoint with ONE field
         * changed: tokenizer_version 2 -> 1, and the CRC and SHA-256
         * recomputed so the file still verifies. A hand-written stub cannot
         * reach this check at all -- it is behind the magic check and the
         * integrity check, so a stub gets rejected for one of those and the
         * message the user actually sees is never exercised. */
        {
            size_t flen = 0;
            int fok = 0;
            char *fb = aster_read_file("models/selftest.bin", &flen, &fok);
            int made = 0;
            if (fb && flen > 36 + 16) {
                fb[16] = 1; fb[17] = 0; fb[18] = 0; fb[19] = 0;   /* tokenizer_version */
                size_t body = flen - 36;
                uint32_t crc = aster_crc32(fb, body);
                unsigned char *p = (unsigned char *)fb + body;
                p[0] = (char)(crc);       p[1] = (char)(crc >> 8);
                p[2] = (char)(crc >> 16); p[3] = (char)(crc >> 24);
                aster_sha256_raw(fb, body, (unsigned char *)fb + body + 4);
                aster_write_file_atomic("models/selftest-v1.bin", fb, flen);
                made = 1;
            }
            free(fb);
            if (!made) aster_warn("could not build the v1 checkpoint fixture");
            err[0] = '\0';
            r = aster_checkpoint_load("models/selftest-v1.bin", NULL, NULL, NULL, NULL, err, sizeof err);
            check(r == NULL, "a byte-tokenizer (v1) checkpoint is refused", &failures);
            if (r) aster_model_free(r);
            else {
                aster_info("  reason: %s", err);
                /* The message must name the actual cause, or the user is sent
                 * to debug the wrong thing. */
                int mentions_retrain = strstr(err, "retrain") != NULL;
                int names_v1 = strstr(err, "byte tokenizer") != NULL;
                int not_damaged = strstr(err, "damaged") == NULL;
                check(mentions_retrain, "the v1 refusal tells the user to retrain", &failures);
                check(names_v1, "the v1 refusal names the tokenizer, not corruption", &failures);
                check(not_damaged, "an intact v1 file is not reported as damaged", &failures);
            }
        }

        /* Missing file. */
        r = aster_checkpoint_load("models/definitely-not-here.bin", NULL, NULL, NULL, NULL, err, sizeof err);
        check(r == NULL, "a missing checkpoint is rejected", &failures);
        if (r) aster_model_free(r);

        remove("models/selftest.bin");
        remove("models/selftest-trunc.bin");
        remove("models/selftest-bad.bin");
        remove("models/selftest-v1.bin");
        aster_model_free(m);
    }

    /* ---- greedy generation is deterministic, and honours the cap ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg, &V);
        AsterModel *m = aster_model_new(&cfg, &V, 5u, 0);
        GenParams gp;
        gen_params_default(&gp);
        gp.max_new_tokens = 20;
        gp.temperature = 0.0f;      /* greedy */
        gp.seed = 3;
        size_t cap = aster_generate_out_cap(m);
        char *a = (char *)aster_xmalloc(cap);
        char *b = (char *)aster_xmalloc(cap);
        int he = 0, lt = 0, tr = 0;
        aster_generate(m, "hello", &gp, a, cap, &he, &lt, &tr);
        aster_generate(m, "hello", &gp, b, cap, &he, &lt, &tr);
        check(strcmp(a, b) == 0, "greedy generation is reproducible for a fixed seed", &failures);
        check(utf8_is_valid(a, strlen(a)), "generated output is valid UTF-8", &failures);
        /* The cap is in TOKENS, so the byte length of the reply is not bounded
         * by max_new_tokens: one token can spell a whole word. The bound that
         * still holds is tokens * max_token_bytes, plus what the UTF-8
         * sanitiser can expand each invalid byte into (U+FFFD is 3 bytes). */
        int mtb = aster_vocab_max_token_bytes(&m->vocab);
        check(strlen(a) <= (size_t)20 * (size_t)mtb * 3,
              "generation respects max_new_tokens", &failures);
        free(a); free(b);
        aster_model_free(m);
    }

    /* ---- generation reads every step it takes, not just the first ----
     *
     * aster_forward computes exactly a->T rows. If the acts are sized to the
     * initial prompt rather than the whole context, every row past the prompt
     * stays at its calloc'd zero, the logits look uniformly 0.00, and argmax
     * returns a control token -- so generation silently stops after one
     * character while the loss on held-out data still looks healthy. Decoding
     * here by hand, with acts sized for the full context, and requiring the two
     * to agree catches that: they cannot agree unless every step is a real
     * forward pass over the real sequence. */
    {
        AsterConfig cfg;
        aster_config_default(&cfg, &V);
        AsterModel *m = aster_model_new(&cfg, &V, 11u, 0);
        /* The output head is tied to the token embedding, so pushing the five
         * control rows far negative makes argmax unable to select them. Without
         * this an untrained model stops at step 0 and the test would never
         * reach a second forward pass, which is exactly where the bug lived.
         * It also pushes every LEARNED MERGE row negative, because under a
         * sub-word vocabulary those ids are ordinary content, not markers, and
         * the test needs generation to run past the first step. */
        for (int id = TOK_FIRST_MERGE; id < cfg.vocab; ++id)
            for (int c = 0; c < cfg.d_model; ++c)
                m->params[m->off.tok_emb + (size_t)id * cfg.d_model + c] = -40.0f;
        for (int id = TOK_SYSTEM; id <= TOK_PAD; ++id)
            for (int c = 0; c < cfg.d_model; ++c)
                m->params[m->off.tok_emb + (size_t)id * cfg.d_model + c] = -40.0f;
        const char *prompt = "are you a robot";
        const int C = cfg.context;
        int max_new = 24;

        /* Reference decode: argmax only, arena sized for the whole context. */
        TokenList pl;
        token_list_init(&pl);
        tok_encode(&m->vocab, prompt, strlen(prompt), &pl);
        uint16_t *seq = (uint16_t *)aster_xmalloc((size_t)C * sizeof(uint16_t));
        int n = 0;
        seq[n++] = TOK_SYSTEM;
        seq[n++] = TOK_USER;
        for (size_t i = 0; i < pl.count && n < C; ++i) seq[n++] = pl.ids[i];
        seq[n++] = TOK_END;
        seq[n++] = TOK_ASSISTANT;
        token_list_free(&pl);
        AsterActs *ra = aster_acts_new(1, C, &cfg);
        uint16_t *ref_ids = (uint16_t *)aster_xmalloc((size_t)max_new * sizeof(uint16_t));
        int rn_ids = 0;
        for (int step = 0; step < max_new && n < C; ++step) {
            ra->T = n;
            if (aster_forward(m, ra, seq, NULL, NULL, NULL) < 0.0) break;
            const float *lg = ra->logits + (size_t)(n - 1) * (size_t)cfg.vocab;
            int best = 0;
            for (int v = 1; v < cfg.vocab; ++v) if (lg[v] > lg[best]) best = v;
            /* Mirror aster_generate's stop rule EXACTLY. END, PAD, and the
             * three turn markers end generation; byte ids and learned merges
             * are both content. An earlier version of this loop used a
             * ceiling test, which matched a bug in aster_generate rather than
             * the intended behaviour -- so the two agreed on empty output and
             * the comparison passed while generation was broken. */
            if (best == TOK_END || best == TOK_PAD) break;
            if (best >= TOK_SYSTEM && best <= TOK_ASSISTANT) break;
            seq[n++] = (uint16_t)best;
            ref_ids[rn_ids++] = (uint16_t)best;
        }
        aster_acts_free(ra);
        free(seq);

        /* Decode the reference ids through the vocabulary, so this compares
         * token choices and detokenization, not two copies of the same bug. */
        size_t refcap = (size_t)max_new * (size_t)aster_vocab_max_token_bytes(&m->vocab) * 3 + 32;
        char *ref = (char *)aster_xmalloc(refcap);
        size_t rn = tok_decode(&m->vocab, ref_ids, (size_t)rn_ids, ref, refcap, 0);
        ref[rn] = '\0';
        free(ref_ids);

        /* Compare like with like: aster_generate runs the raw bytes through the
         * UTF-8 sanitiser before returning, and an untrained model emits high
         * bytes that the sanitiser replaces. Sanitise the reference too, or the
         * comparison passes trivially with both sides empty. */
        char *ref_s = (char *)aster_xmalloc(refcap);
        size_t ref_sn = utf8_sanitize(ref, rn, ref_s, refcap);

        GenParams gp;
        gen_params_default(&gp);
        gp.max_new_tokens = max_new;
        gp.temperature = 0.0f;
        gp.seed = 1;
        size_t cap = aster_generate_out_cap(m);
        char *got = (char *)aster_xmalloc(cap);
        int he = 0, lt = 0, tr = 0;
        size_t gn = aster_generate(m, prompt, &gp, got, cap, &he, &lt, &tr);

        /* Assert on the raw decode length: that is the loop actually running
         * the requested number of steps, which is the thing that was broken. */
        check(rn_ids == max_new,
              "generation keeps decoding for every requested step", &failures);
        /* Non-empty output. A stop rule that treats a byte id as a turn marker
         * stops after zero tokens and still satisfies every other check here,
         * so the one thing that actually distinguishes "generated" from
         * "stopped immediately" is asserted directly. */
        check(gn > 0 && strlen(got) > 0,
              "generation produces output rather than stopping at step 0", &failures);
        check(gn == ref_sn && strcmp(ref_s, got) == 0,
              "generation matches a reference decode over the whole context", &failures);
        free(ref); free(ref_s); free(got);
        aster_model_free(m);
    }

    /* ---- JSON helpers ---- */
    {
        char out[256], err[160];
        const char *ok_body = "{\"message\":\"hi\\nthere\",\"n\":5}";
        int rc = json_get_string(ok_body, strlen(ok_body), "message", out, sizeof out, err, sizeof err);
        check(rc == 0 && strcmp(out, "hi\nthere") == 0, "JSON string decoding with escapes", &failures);

        const char *cut = "{\"message\":\"a\",";
        rc = json_get_string(cut, strlen(cut), "message", out, sizeof out, err, sizeof err);
        check(rc == -1, "truncated JSON is rejected", &failures);

        const char *trail = "{\"message\":\"a\"} trailing";
        rc = json_get_string(trail, strlen(trail), "message", out, sizeof out, err, sizeof err);
        check(rc == -1, "trailing content after the object is rejected", &failures);

        const char *unicode = "{\"message\":\"caf\\u00e9\"}";
        rc = json_get_string(unicode, strlen(unicode), "message", out, sizeof out, err, sizeof err);
        check(rc == 0 && strcmp(out, "caf\xC3\xA9") == 0, "JSON \\u escape becomes UTF-8", &failures);

        char esc[64];
        check(json_escape("a\"b\\c\nd", esc, sizeof esc) == 0 && strcmp(esc, "a\\\"b\\\\c\\nd") == 0,
              "JSON escaping round trip", &failures);
    }

    /* ---- chat loss mask is aligned to the shifted target ----
     * dataset_gather() predicts stream[i+1] at position i, so the first answer
     * TOKEN is trained by the ASSISTANT marker position, not by the first token
     * of the answer. Getting this backwards silently trains the model never to
     * open a reply. Under a sub-word vocabulary this matters more, not less:
     * the unit that would be dropped is a whole leading word rather than a
     * letter, so the model would learn to start every reply mid-word. */
    {
        const char *tmp = "aster_selftest_chat.jsonl";
        static const char *jsonl =
            "{\"role\":\"system\",\"content\":\"\"}\n"
            "{\"role\":\"user\",\"content\":\"hi\"}\n"
            "{\"role\":\"assistant\",\"content\":\"hello\"}\n";
        FILE *f = fopen(tmp, "wb");
        if (!f) {
            aster_warn("could not write %s; skipping the loss-mask test", tmp);
        } else {
            fwrite(jsonl, 1, strlen(jsonl), f);
            fclose(f);

            const char *paths[1] = { tmp };
            DataSpec spec;
            spec.paths = paths;
            spec.n_paths = 1;
            spec.mode = ASTER_DATA_CHAT;
            spec.block = 64;
            spec.max_examples = 0;
            spec.v = &V0;   /* byte fallback, so the token count is checkable */
            char err[192];
            AsterDataset *d = dataset_build(&spec, err, sizeof err);

            /* window: SYSTEM USER "hi" END ASSISTANT "hello" END
             *                                     ^ first answer token target */
            int found = 0, good = 0;
            if (d && d->n_win == 1) {
                const AsterWindow *e = &d->win[0];
                for (int i = 0; i < e->len; ++i) {
                    if (d->stream[e->start + (size_t)i] != TOK_ASSISTANT) continue;
                    found = 1;
                    /* The marker trains the first answer token, the prompt
                     * above it is masked, the token after the marker is
                     * trained too, and the closing END (which has no
                     * successor) is not. */
                    uint16_t first = (i + 1 < e->len) ? d->stream[e->start + (size_t)i + 1] : 0;
                    if (e->w && i > 0 && e->w[i] > 0.0f && e->w[i - 1] == 0.0f &&
                        i + 1 < e->len && e->w[i + 1] > 0.0f &&
                        first == (uint16_t)'h' &&
                        e->w[e->len - 1] == 0.0f)
                        good = 1;
                    break;
                }
            } else {
                aster_warn("loss-mask test: dataset did not build (%s)", d ? "wrong window count" : err);
            }
            check(found && good,
                  "chat loss mask trains the first answer token", &failures);

            dataset_free(d);
            remove(tmp);
        }
    }
    {
        /* The same mask must hold under a learned vocabulary, where the first
         * answer token is a MERGE covering all five letters rather than one
         * byte. A mask written in bytes would drop it, and the whole reply
         * would become untrained. */
        const char *tmp = "aster_selftest_chat_bpe.jsonl";
        static const char *jsonl =
            "{\"role\":\"system\",\"content\":\"\"}\n"
            "{\"role\":\"user\",\"content\":\"hi\"}\n"
            "{\"role\":\"assistant\",\"content\":\"hello\"}\n";
        FILE *f = fopen(tmp, "wb");
        if (!f) {
            aster_warn("could not write %s; skipping the sub-word mask test", tmp);
        } else {
            fwrite(jsonl, 1, strlen(jsonl), f);
            fclose(f);
            const char *paths[1] = { tmp };
            DataSpec spec;
            spec.paths = paths;
            spec.n_paths = 1;
            spec.mode = ASTER_DATA_CHAT;
            spec.block = 64;
            spec.max_examples = 0;
            spec.v = &V;
            char err[192];
            AsterDataset *d = dataset_build(&spec, err, sizeof err);
            int ok = 0;
            if (d && d->n_win == 1) {
                const AsterWindow *e = &d->win[0];
                for (int i = 0; i < e->len; ++i) {
                    if (d->stream[e->start + (size_t)i] != TOK_ASSISTANT) continue;
                    if (i + 1 < e->len && e->w && e->w[i] > 0.0f && i > 0 && e->w[i - 1] == 0.0f) {
                        /* Whatever token follows the marker must itself be
                         * trained, and must spell the start of "hello". */
                        uint16_t first = d->stream[e->start + (size_t)i + 1];
                        const char *spell = NULL;
                        size_t sl = 0;
                        if (tok_spelling(&V, first, &spell, &sl) == 0 && sl <= 5 &&
                            memcmp(spell, "hello", sl) == 0)
                            ok = 1;
                    }
                    break;
                }
            }
            check(ok, "the loss mask still trains the first answer token under merges", &failures);
            dataset_free(d);
            remove(tmp);
        }
    }

    /* ---- bits-per-byte is TOTAL nats over TOTAL bytes ----
     *
     * The tempting definition -- a byte-weighted AVERAGE of per-token
     * cross-entropies -- double-counts long tokens, because a token's
     * cross-entropy already covers every byte it spans. That version reported
     * 5.26 nats/byte here where the correct figure is 1.63, and the error grew
     * with the compression ratio, so it got worse exactly as the tokenizer
     * improved.
     *
     * The invariant is: nats_per_byte == nats_per_token * tokens / bytes. It
     * holds only if both figures share one numerator, and it is checkable from
     * the reported values alone. It needs a vocabulary with MIXED token lengths
     * to have any teeth -- with a byte vocabulary every target is 1 byte and
     * both definitions coincide, so the test would pass either way. */
    {
        const char *tmp = "aster_selftest_bpb.txt";
        FILE *f = fopen(tmp, "wb");
        int ok = 0;
        if (f) {
            for (int i = 0; i < 12; ++i)
                fprintf(f, "the model is honest and the model is small. the quick brown fox.\n");
            fclose(f);
            const char *paths[1] = { tmp };
            DataSpec spec;
            memset(&spec, 0, sizeof spec);
            spec.paths = paths;
            spec.n_paths = 1;
            spec.mode = ASTER_DATA_TEXT;
            spec.block = 64;
            spec.v = &V;
            char err[192];
            AsterDataset *d = dataset_build(&spec, err, sizeof err);
            if (d) {
                AsterConfig cfg;
                aster_config_default(&cfg, &V);
                AsterModel *m = aster_model_new(&cfg, &V, 3u, 0);
                AsterLoss L;
                memset(&L, 0, sizeof L);
                if (dataset_eval_loss_full(m, d, 4, &L) > 0.0 && L.bytes > 0) {
                    double expect = L.nats_per_token * (double)L.tokens / (double)L.bytes;
                    ok = fabs(L.nats_per_byte - expect) < 1e-9 * (fabs(expect) + 1.0);
                    aster_info("  %.4f nats/token, %zu tokens, %zu bytes -> %.4f nats/byte "
                               "(%.3f bits/byte)", L.nats_per_token, L.tokens, L.bytes,
                               L.nats_per_byte, L.bits_per_byte);
                    /* A model that knows nothing about bytes spends log2(256) =
                     * 8 bits per byte. Anything far above that means the figure
                     * is being computed wrong, not that the model is bad. */
                    if (L.bits_per_byte > 8.0)
                        aster_warn("  bits/byte is %.2f, worse than a uniform byte guess (8.0). "
                                   "That is possible for an untrained model, but check the "
                                   "arithmetic if a trained model reports it.", L.bits_per_byte);
                }
                aster_model_free(m);
            }
            dataset_free(d);
            remove(tmp);
        }
        check(ok, "bits/byte is total nats over total bytes, not a weighted average", &failures);
    }

    /* ---- nats -> bits is a division by ln 2, not an exponentiation ----
     *
     * `exp(nats/ln2)` is monotonically increasing, so it does not change which
     * checkpoint is selected -- it only makes every printed number wrong. And
     * wrong how: 1.4 nats/byte became 2.07 bits/byte (correct) versus
     * exp(2.07) = 7.9. Nothing above can see it, because nats/byte is
     * separately checked and the two were believed to be the same quantity.
     * The identity is a one-liner, so check it directly.
     *
     * Anchored on a known value rather than on the loss: log2(256) is exactly
     * 8 bits per byte, so an untrained byte-level model must report ~8.0, and
     * exp() would report e^8 = 2981. */
    {
        int ok = 0;
        double nats = 1.4351;
        AsterLoss L;
        memset(&L, 0, sizeof L);
        L.nats_per_byte = nats;
        L.bits_per_byte = nats / log(2.0);
        ok = fabs(L.bits_per_byte - 2.0704) < 5e-4
             && L.bits_per_byte < L.nats_per_byte * 2.0   /* dividing, not growing */
             && fabs(log(256.0) / log(2.0) - 8.0) < 1e-12;  /* log2(256) == 8 */
        aster_info("  %.4f nats/byte is %.4f bits/byte (exp would say %.1f)",
                   nats, L.bits_per_byte, exp(nats / log(2.0)));
        check(ok, "nats -> bits divides by ln 2 rather than exponentiating", &failures);
    }

    /* ---- analytic gradients match a finite difference of the loss ----
     * The manual backward pass is the part of this program most easily wrong,
     * and a wrong gradient still trains: the loss falls while the model learns
     * token frequencies instead of conditioning on the prompt. Nothing else
     * here would notice, so it is checked directly against the forward pass.
     *
     * Only gradients within 1e-2 of the largest are compared. The loss is
     * accumulated in float32, so f(x) carries about 1e-6 of absolute error and
     * the finite difference of a tiny gradient is pure rounding: measured on
     * this code, gradients above that cut agree to 0.7% while those below 1e-4
     * of the maximum can be 40% off from float32 noise alone. Checking the
     * negligible ones would mean asserting that rounding errors match. */
    {
        /* one layer and two, because the residual stream is indexed with an
         * L+1 stride and a single-layer model cannot catch a wrong stride */
        static const int shape[2][3] = { {1, 8, 2}, {2, 8, 2} };
        for (int ci = 0; ci < 2; ++ci) {
            AsterConfig gc;
            aster_config_default(&gc, &V0);
            gc.n_layer = shape[ci][0];
            gc.d_model = shape[ci][1];
            gc.n_head  = shape[ci][2];
            gc.d_ff    = 2 * gc.d_model;
            gc.context = 8;
            char err[160];
            if (aster_config_validate(&gc, err, sizeof err) != 0) {
                aster_warn("gradient check: %s", err);
                check(0, "analytic gradients match a finite difference", &failures);
                continue;
            }

            AsterModel *gm = aster_model_new(&gc, &V0, 7u, 1);
            AsterActs  *ga = aster_acts_new(1, 4, &gc);
            const uint16_t gx[4] = {1, 5, 9, 20};
            const uint16_t gy[4] = {5, 9, 20, 42};
            const float    gw[4] = {1.0f, 1.0f, 1.0f, 1.0f};

            double l0 = 0.0;
            aster_forward(gm, ga, gx, gy, gw, &l0);
            memset(gm->grads, 0, sizeof(float) * (size_t)gm->off.total);
            aster_backward(gm, ga, gx, gy, gw);

            const int n = gm->off.total;
            double *num = (double *)malloc(sizeof(double) * (size_t)n);
            int ok = 1, compared = 0;
            if (!num) {
                check(0, "analytic gradients match a finite difference", &failures);
            } else {
                double maxnum = 0.0;
                for (int i = 0; i < n; ++i) {
                    num[i] = gc_numeric(gm, ga, gx, gy, gw, i);
                    double a = fabs(num[i]);
                    if (a > maxnum) maxnum = a;
                }
                double worst = 0.0;
                int worst_i = -1;
                for (int i = 0; i < n; ++i) {
                    if (fabs(num[i]) < 1e-2 * maxnum) continue;
                    ++compared;
                    double r = fabs((double)gm->grads[i] - num[i]) / fabs(num[i]);
                    if (r > worst) { worst = r; worst_i = i; }
                    if (r > 0.02) ok = 0;
                }
                if (worst_i >= 0 && !ok)
                    aster_warn("  %d layers: param %d analytic %+.6e numeric %+.6e",
                               gc.n_layer, worst_i, gm->grads[worst_i], num[worst_i]);
                aster_info("  %d-layer check: %d gradients compared, worst relative error %.4f",
                           gc.n_layer, compared, worst);
                free(num);
            }
            char what[96];
            snprintf(what, sizeof what,
                     "analytic gradients match a finite difference (%d layer%s)",
                     gc.n_layer, gc.n_layer == 1 ? "" : "s");
            check(ok, what, &failures);

            aster_acts_free(ga);
            aster_model_free(gm);
        }
    }

    st_vocab_free(&V);
    st_vocab_free(&V0);

    puts("");
    if (failures == 0) { aster_info("self-test passed"); return 0; }
    aster_fail("self-test found %d failure(s)", failures);
    return 1;
}

/* ------------------------------------------------------------- subcommands */

static int cmd_train(int argc, char **argv) {
    Args a;
    parse_args(argc, argv, &a, 2);
    if (a.n_data < 1) {
        fprintf(stderr, "error: --data is required. Point it at your own UTF-8 text file(s).\n");
        return 2;
    }
    TrainConfig tc;
    train_config_default(&tc);
    if (a.has_seed) tc.seed = a.seed;
    if (a.steps >= 0) tc.steps = a.steps;
    if (a.batch > 0) tc.batch = a.batch;
    if (a.log_every > 0) tc.log_every = a.log_every;
    if (a.has_lr) tc.lr = a.lr;
    if (tc.steps < 1) { fprintf(stderr, "error: --steps must be at least 1\n"); return 2; }
    if (tc.batch < 1) { fprintf(stderr, "error: --batch must be at least 1\n"); return 2; }
    if (tc.batch > 512) { fprintf(stderr, "error: --batch above 512 will be very slow on a CPU\n"); return 2; }
    if (tc.log_every < 1) tc.log_every = 1;

    DataSpec ts, vs;
    memset(&ts, 0, sizeof ts);
    memset(&vs, 0, sizeof vs);
    ts.paths = a.data;   ts.n_paths = a.n_data;   ts.block = a.block;
    vs.paths = a.valid; vs.n_paths = a.n_valid;   vs.block = a.block;
    ts.mode = vs.mode = (strcmp(a.mode, "chat") == 0) ? ASTER_DATA_CHAT : ASTER_DATA_TEXT;

    char err[512];
    puts("");
    /* Learn (or load) the vocabulary BEFORE the config, because the config's
     * vocab size is the vocabulary's. */
    AsterVocab *v = vocab_resolve(&a, &ts, err, sizeof err);
    if (!v) { fprintf(stderr, "error: %s\n", err); return 1; }
    ts.v = v;
    vs.v = v;

    /* Writing the vocabulary next to the checkpoint is not required -- the
     * checkpoint carries its own merge table -- but it is how a user gets a
     * tokenizer they can inspect, and how a second run reproduces the first. */
    if (!a.vocab) {
        char vpath[512];
        snprintf(vpath, sizeof vpath, "%s.vocab", a.out);
        if (aster_vocab_save(v, vpath, err, sizeof err) != 0)
            aster_warn("could not write %s: %s (the checkpoint still carries the merges)", vpath, err);
        else
            aster_info("wrote %s", vpath);
    }

    AsterConfig mcfg;
    aster_config_default(&mcfg, v);
    int rc = aster_train(&tc, &mcfg, &ts, a.n_valid ? &vs : NULL, a.out, err, sizeof err);
    aster_vocab_free(v);
    free(v);
    if (rc != 0) {
        aster_fail("training failed: %s", err);
        return 1;
    }
    return 0;
}

static int load_model(const char *path, AsterModel **out, uint32_t *step, uint32_t *seed) {
    char err[256];
    *out = aster_checkpoint_load(path, NULL, NULL, step, seed, err, sizeof err);
    if (!*out) {
        fprintf(stderr, "error: %s\n", err);
        fprintf(stderr, "hint : train one with  aster train --data <your text file> "
                        "--out %s --seed 1234\n", path);
        return -1;
    }
    return 0;
}

static int cmd_generate(int argc, char **argv) {
    Args a;
    parse_args(argc, argv, &a, 2);
    if (!a.model) { fprintf(stderr, "error: --model is required\n"); return 2; }
    AsterModel *m;
    uint32_t step = 0, seed = 0;
    if (load_model(a.model, &m, &step, &seed) != 0) return 1;

    GenParams gp;
    gen_params_default(&gp);
    if (a.has_max_new) gp.max_new_tokens = a.max_new;
    if (a.has_temp) gp.temperature = a.temp;
    if (a.has_seed) gp.seed = a.seed;
    else gp.seed = 1234u;
    gp.top_k = a.top_k;
    gp.system = a.system;

    aster_info("model %s: %d parameters, %d-token context, %d tokens in the vocabulary "
               "(%d merges, %.2f bytes/token), trained %u step(s)",
               file_name_of(a.model), m->off.total, m->cfg.context, m->cfg.vocab,
               m->vocab.n_merges, m->vocab.bytes_per_token, step);
    int budget = aster_prompt_budget_for(m, gp.max_new_tokens, a.system);
    aster_info("sampling: %s, temperature %.2f, top_k %d, seed %u; at most %d prompt token(s) fit",
               gp.temperature <= 0.0f ? "greedy" : "stochastic",
               (double)gp.temperature, gp.top_k, gp.seed, budget);
    if (a.prompt) {
        TokenList pl;
        token_list_init(&pl);
        int enc = tok_encode(&m->vocab, a.prompt, strlen(a.prompt), &pl);
        if (enc != 0) aster_warn("the prompt could not be encoded and was skipped");
        else if ((int)pl.count > budget)
            aster_warn("the prompt is %zu tokens but only %d fit, so its start will be dropped",
                       pl.count, budget);
        token_list_free(&pl);
    }
    puts("");

    size_t cap = aster_generate_out_cap(m);
    char *out = (char *)aster_xmalloc(cap);
    int hit_end = 0, left_turn = 0, truncated = 0;
    aster_generate(m, a.prompt ? a.prompt : "", &gp, out, cap, &hit_end, &left_turn, &truncated);
    puts("--- generated text (raw model output) ---");
    puts(out);
    puts("--- end ---");
    if (hit_end) aster_info("stopped on the END marker");
    if (left_turn) aster_warn("the model started a new turn instead of finishing its reply");
    if (truncated) aster_warn("the context filled up before the reply was finished");
    aster_info("this is a %d-parameter experimental model; the text above is a statistical "
               "continuation, not a verified answer", m->off.total);
    free(out);
    aster_model_free(m);
    return 0;
}

static int cmd_serve(int argc, char **argv) {
    Args a;
    parse_args(argc, argv, &a, 2);
    AsterServer s;
    memset(&s, 0, sizeof s);
    s.max_new_tokens = a.has_max_new ? a.max_new : ASTER_DEFAULT_MAX_NEW;
    s.temperature = a.has_temp ? a.temp : 0.0f;
    s.top_k = a.top_k;
    s.system = a.system;
    s.seed = a.has_seed ? a.seed : 1234u;

    if (s.max_new_tokens < 1) s.max_new_tokens = 1;
    if (s.max_new_tokens > 512) { fprintf(stderr, "error: --max-new-tokens above 512 is refused\n"); return 2; }
    if (s.top_k < 0) s.top_k = 0;
    if (s.temperature < 0.0f) s.temperature = 0.0f;

    if (a.model) {
        if (load_model(a.model, &s.model, &s.step, &s.seed) != 0) return 1;
        snprintf(s.name, sizeof s.name, "%s", file_name_of(a.model));
    } else {
        aster_warn("no --model was given. The server will start, report that the model is "
                   "not trained, and answer chat requests with 503 until you train one.");
        snprintf(s.name, sizeof s.name, "%s", "none");
    }
    puts("");
    aster_serve(&s, a.host, a.port);
    return 0;
}

static int cmd_eval(int argc, char **argv) {
    Args a;
    parse_args(argc, argv, &a, 2);
    if (!a.model) { fprintf(stderr, "error: --model is required\n"); return 2; }
    if (a.n_data < 1) { fprintf(stderr, "error: --data is required\n"); return 2; }
    AsterModel *m;
    uint32_t step = 0, seed = 0;
    if (load_model(a.model, &m, &step, &seed) != 0) return 1;

    DataSpec vs;
    memset(&vs, 0, sizeof vs);
    vs.paths = a.data;
    vs.n_paths = a.n_data;
    vs.block = a.block;
    vs.mode = (strcmp(a.mode, "chat") == 0) ? ASTER_DATA_CHAT : ASTER_DATA_TEXT;
    vs.v = &m->vocab;   /* the checkpoint's own merges; the model must use them */

    char err[512];
    AsterDataset *d = dataset_build(&vs, err, sizeof err);
    if (!d) { aster_fail("%s", err); return 1; }

    AsterLoss L;
    double loss = dataset_eval_loss_full(m, d, 8, &L);
    if (loss < 0.0) { dataset_free(d); aster_fail("evaluation failed"); return 1; }

    aster_info("model:    %s (%d parameters, step %u, seed %u)",
               file_name_of(a.model), m->off.total, step, seed);
    aster_info("tokenizer: bpe-2, %d merges, %d tokens, %.3f bytes/token",
               m->vocab.n_merges, m->cfg.vocab, m->vocab.bytes_per_token);
    aster_info("data:     %d window(s), %zu tokens, hash %s", d->n_win, d->tokens, d->hash);
    aster_info("");
    aster_info("held-out loss");
    /* nats/byte and bits/byte are the same measurement in two units, so they
     * are printed as one line with the conversion shown rather than as two
     * rows that look like two different results. */
    aster_info("  %.4f bits/byte  <- the figure to compare across tokenizers", L.bits_per_byte);
    aster_info("  (that is %.4f nats/byte; nats/byte and bits/byte are one number in two units,",
               L.nats_per_byte);
    aster_info("   related by a division by ln 2, not a change of definition)");
    aster_info("  %.4f nats/token  (perplexity %.2f; comparable only against a model using",
               L.nats_per_token, L.token_ppl);
    aster_info("   the SAME vocabulary, so do not compare it to the number above)");
    aster_info("  over %zu scored tokens covering %zu bytes (%.2f bytes/token)",
               L.weight, L.bytes, (double)L.bytes / (double)(L.weight ? L.weight : 1));
    aster_info("");
    aster_info("Compare Aster versions on the SAME evaluation set before claiming improvement,");
    aster_info("and compare bits/byte, not nats/token, across a tokenizer change.");
    dataset_free(d);
    aster_model_free(m);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_help();
        return 2;
    }
    const char *cmd = argv[1];
    if (!strcmp(cmd, "help") || !strcmp(cmd, "--help") || !strcmp(cmd, "-h")) { print_help(); return 0; }
    if (!strcmp(cmd, "selftest")) return cmd_selftest();
    if (!strcmp(cmd, "train"))    return cmd_train(argc, argv);
    if (!strcmp(cmd, "vocab"))    return cmd_vocab(argc, argv);
    if (!strcmp(cmd, "generate")) return cmd_generate(argc, argv);
    if (!strcmp(cmd, "serve"))    return cmd_serve(argc, argv);
    if (!strcmp(cmd, "eval"))     return cmd_eval(argc, argv);

    fprintf(stderr, "error: unknown command \"%s\"\n\n", cmd);
    print_help();
    return 2;
}
