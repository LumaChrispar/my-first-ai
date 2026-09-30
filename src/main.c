/* main.c - command-line dispatch.
 *
 * Commands: train, generate, serve, eval, selftest, help.
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
    puts("  .\\aster.exe train --mode chat --steps 600 `");
    puts("      --data data/demo_chat.jsonl --validation data/demo_valid.jsonl `");
    puts("      --out models/aster-small.bin --seed 1234");
    puts("  .\\aster.exe generate --model models/aster-small.bin --prompt \"Who are you?\"");
    puts("  .\\aster.exe serve --model models/aster-small.bin --port 8080");
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
    const char *prompt;
    const char *system;
    const char *host;
    const char *mode;
    int    port;
    int    steps, batch, block, log_every, max_new, top_k, has_port, has_max_new;
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

#define NEXT() do { if (i + 1 >= argc) aster_fail("%s needs a value", argv[i]); } while (0)
    for (int i = start; i < argc; ++i) {
        const char *k = argv[i];
        if      (!strcmp(k, "--data"))      { NEXT(); if (a->n_data >= MAX_DATA_FILES) aster_fail("too many --data files"); a->data[a->n_data++] = argv[++i]; }
        else if (!strcmp(k, "--validation")){ NEXT(); if (a->n_valid >= MAX_DATA_FILES) aster_fail("too many --validation files"); a->valid[a->n_valid++] = argv[++i]; }
        else if (!strcmp(k, "--out"))       { NEXT(); a->out = argv[++i]; }
        else if (!strcmp(k, "--model"))     { NEXT(); a->model = argv[++i]; }
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

static int cmd_selftest(void) {
    int failures = 0;
    aster_info("Aster self-test");
    puts("");

    /* ---- tokenizer round trips ---- */
    {
        const char *ascii = "The quick brown fox jumps over the lazy dog. 0123456789";
        TokenList tl;
        token_list_init(&tl);
        tok_encode(ascii, strlen(ascii), &tl);
        char back[256];
        tok_decode(tl.ids, tl.count, back, sizeof back, 0);
        check(strcmp(ascii, back) == 0, "ASCII encode/decode round trip", &failures);
        token_list_free(&tl);
    }
    {
        /* Valid multi-byte UTF-8, written out explicitly so the source file
         * encoding cannot affect the test. */
        const char utf8[] = "caf\xC3\xA9 na\xC3\xAFve \xE2\x9C\x93 \xF0\x9F\x9A\x80 done";
        TokenList tl;
        token_list_init(&tl);
        tok_encode(utf8, sizeof utf8 - 1, &tl);
        char back[256];
        tok_decode(tl.ids, tl.count, back, sizeof back, 0);
        int ok = (strcmp(utf8, back) == 0) && utf8_is_valid(back, strlen(back));
        check(ok, "UTF-8 encode/decode round trip", &failures);
        token_list_free(&tl);
    }
    {
        /* Special markers must never collide with byte tokens. */
        int ok = TOK_SYSTEM > 255 && TOK_USER > 255 && TOK_ASSISTANT > 255 &&
                 TOK_END > 255 && TOK_PAD == 260 && ASTER_VOCAB == 261;
        check(ok, "special markers are distinct and above the byte range", &failures);
    }
    {
        /* A long input must be reported, not written past the buffer. */
        char big[8192];
        memset(big, 'x', sizeof big - 1);
        big[sizeof big - 1] = '\0';
        TokenList tl;
        token_list_init(&tl);
        tok_encode(big, sizeof big - 1, &tl);
        char small[16];
        size_t n = tok_decode(tl.ids, tl.count, small, sizeof small, 0);
        int ok = (tl.count == sizeof big - 1) && (n == sizeof small - 1) && small[sizeof small - 1] == '\0';
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

    /* ---- parameter count matches the documented architecture ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg);
        int n = aster_param_count(&cfg);
        aster_info("default architecture parameter count: %d (%.4f million)", n, n / 1e6);
        /* tok_emb + pos_emb + 2 layers + final norm, from the layout:
         * 261*64 + 128*64 + 2*(2*64 + 4*(64*64) + 2*64 + 64*256+256 + 256*64) + 2*64 */
        int expect = 261 * 64 + 128 * 64 +
                     2 * (2 * 64 + 4 * (64 * 64) + 2 * 64 + 64 * 256 + 256 + 256 * 64) +
                     2 * 64;
        check(n == expect, "parameter count matches the documented layout", &failures);
    }

    /* ---- forward pass produces finite values and a loss near ln(vocab) ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg);
        AsterModel *m = aster_model_new(&cfg, 7u, 1);
        const int B = 2, T = 16;
        AsterActs *a = aster_acts_new(B, T, &cfg);
        uint16_t *x = (uint16_t *)aster_xmalloc((size_t)B * T * sizeof(uint16_t));
        uint16_t *y = (uint16_t *)aster_xmalloc((size_t)B * T * sizeof(uint16_t));
        float *w = (float *)aster_xmalloc((size_t)B * T * sizeof(float));
        for (int i = 0; i < B * T; ++i) { x[i] = (uint16_t)(i * 7 % 256); y[i] = (uint16_t)((i * 13) % 256); w[i] = 1.0f; }
        double loss = aster_forward(m, a, x, y, w, NULL);
        double expect0 = log((double)ASTER_VOCAB);
        int ok = isfinite(loss) && fabs(loss - expect0) < 0.5;
        check(ok, "untrained forward pass is finite and near ln(vocab)", &failures);
        aster_info("  untrained loss %.4f, ln(261) = %.4f", loss, expect0);

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

        free(x); free(y); free(w); free(w0);
        aster_acts_free(a);
        aster_model_free(m);
    }

    /* ---- out-of-range token ids are rejected, not read out of bounds ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg);
        AsterModel *m = aster_model_new(&cfg, 7u, 0);
        AsterActs *a = aster_acts_new(1, 8, &cfg);
        uint16_t bad[8];
        for (int i = 0; i < 8; ++i) bad[i] = ASTER_VOCAB;   /* one past the end */
        double r = aster_forward(m, a, bad, NULL, NULL, NULL);
        check(r == -1.0, "an out-of-range token id is rejected", &failures);
        aster_acts_free(a);
        aster_model_free(m);
    }

    /* ---- checkpoint round trip, plus rejection of corrupt files ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg);
        AsterModel *m = aster_model_new(&cfg, 99u, 0);
        for (int i = 0; i < m->off.total; ++i) m->params[i] = (float)i * 1e-4f;
        aster_checkpoint_save(m, "models/selftest.bin", NULL, NULL, 42u, 99u, 80u, 0.0f);

        char err[256];
        uint32_t step = 0, seed = 0;
        AsterModel *r = aster_checkpoint_load("models/selftest.bin", NULL, NULL, &step, &seed, err, sizeof err);
        int ok = (r != NULL) && step == 42 && seed == 99 && r->off.total == m->off.total;
        for (int i = 0; ok && i < m->off.total; ++i) if (r->params[i] != m->params[i]) { ok = 0; break; }
        check(ok, "checkpoint save/load round trip is exact", &failures);
        aster_model_free(r);

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

        /* Missing file. */
        r = aster_checkpoint_load("models/definitely-not-here.bin", NULL, NULL, NULL, NULL, err, sizeof err);
        check(r == NULL, "a missing checkpoint is rejected", &failures);
        if (r) aster_model_free(r);

        remove("models/selftest.bin");
        remove("models/selftest-trunc.bin");
        remove("models/selftest-bad.bin");
        aster_model_free(m);
    }

    /* ---- greedy generation is deterministic, and honours the cap ---- */
    {
        AsterConfig cfg;
        aster_config_default(&cfg);
        AsterModel *m = aster_model_new(&cfg, 5u, 0);
        GenParams gp;
        gen_params_default(&gp);
        gp.max_new_tokens = 20;
        gp.temperature = 0.0f;      /* greedy */
        gp.seed = 3;
        char a[512], b[512];
        int he = 0, tr = 0;
        aster_generate(m, "hello", &gp, a, sizeof a, &he, &tr);
        aster_generate(m, "hello", &gp, b, sizeof b, &he, &tr);
        check(strcmp(a, b) == 0, "greedy generation is reproducible for a fixed seed", &failures);
        check(utf8_is_valid(a, strlen(a)), "generated output is valid UTF-8", &failures);
        check(strlen(a) <= 20, "generation respects max_new_tokens", &failures);
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
        aster_config_default(&cfg);
        AsterModel *m = aster_model_new(&cfg, 11u, 0);
        /* The output head is tied to the token embedding, so pushing the five
         * control rows far negative makes argmax unable to select them. Without
         * this an untrained model stops at step 0 and the test would never
         * reach a second forward pass, which is exactly where the bug lived. */
        for (int id = TOK_SYSTEM; id <= TOK_PAD; ++id)
            for (int c = 0; c < cfg.d_model; ++c)
                m->params[m->off.tok_emb + (size_t)id * cfg.d_model + c] = -40.0f;
        const char *prompt = "are you a robot";
        const int C = cfg.context;
        int max_new = 24;

        /* Reference decode: argmax only, arena sized for the whole context. */
        uint16_t *seq = (uint16_t *)aster_xmalloc((size_t)C * sizeof(uint16_t));
        int n = 0;
        seq[n++] = TOK_SYSTEM;
        seq[n++] = TOK_USER;
        for (const char *p = prompt; *p && n < C; ++p) seq[n++] = (uint16_t)(unsigned char)*p;
        seq[n++] = TOK_END;
        seq[n++] = TOK_ASSISTANT;
        AsterActs *ra = aster_acts_new(1, C, &cfg);
        char ref[512];
        size_t rn = 0;
        for (int step = 0; step < max_new && n < C; ++step) {
            ra->T = n;
            if (aster_forward(m, ra, seq, NULL, NULL, NULL) < 0.0) break;
            const float *lg = ra->logits + (size_t)(n - 1) * (size_t)cfg.vocab;
            int best = 0;
            for (int v = 1; v < cfg.vocab; ++v) if (lg[v] > lg[best]) best = v;
            if (best > ASTER_BYTE_MAX) break;
            seq[n++] = (uint16_t)best;
            ref[rn++] = (char)(unsigned char)best;
        }
        ref[rn] = '\0';
        aster_acts_free(ra);
        free(seq);

        /* Compare like with like: aster_generate runs the raw bytes through the
         * UTF-8 sanitiser before returning, and an untrained model emits high
         * bytes that the sanitiser replaces. Sanitise the reference too, or the
         * comparison passes trivially with both sides empty. */
        char ref_s[512];
        size_t ref_sn = utf8_sanitize(ref, rn, ref_s, sizeof ref_s);

        GenParams gp;
        gen_params_default(&gp);
        gp.max_new_tokens = max_new;
        gp.temperature = 0.0f;
        gp.seed = 1;
        char got[512];
        int he = 0, tr = 0;
        size_t gn = aster_generate(m, prompt, &gp, got, sizeof got, &he, &tr);

        /* Assert on the raw decode length: that is the loop actually running
         * the requested number of steps, which is the thing that was broken. */
        check(rn == (size_t)max_new,
              "generation keeps decoding for every requested step", &failures);
        check(gn == ref_sn && strcmp(ref_s, got) == 0,
              "generation matches a reference decode over the whole context", &failures);
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
     * byte is trained by the ASSISTANT marker position, not by the first byte.
     * Getting this backwards silently trains the model never to open a reply. */
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
            char err[192];
            AsterDataset *d = dataset_build(&spec, err, sizeof err);

            /* window: SYSTEM USER "hi" END ASSISTANT "hello" END
             *                                     ^ first answer byte target */
            int found = 0, good = 0;
            if (d && d->n_win == 1) {
                const AsterWindow *e = &d->win[0];
                for (int i = 0; i < e->len; ++i) {
                    if (d->stream[e->start + (size_t)i] != TOK_ASSISTANT) continue;
                    found = 1;
                    /* marker trains the first byte, the prompt above it is
                     * masked, the byte after the marker is trained too, and
                     * the closing END (which has no successor) is not. */
                    if (e->w && i > 0 && e->w[i] > 0.0f && e->w[i - 1] == 0.0f &&
                        i + 1 < e->len && e->w[i + 1] > 0.0f &&
                        e->w[e->len - 1] == 0.0f)
                        good = 1;
                    break;
                }
            } else {
                aster_warn("loss-mask test: dataset did not build (%s)", d ? "wrong window count" : err);
            }
            check(found && good,
                  "chat loss mask trains the first answer byte", &failures);

            dataset_free(d);
            remove(tmp);
        }
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
            aster_config_default(&gc);
            gc.n_layer = shape[ci][0];
            gc.d_model = shape[ci][1];
            gc.n_head  = shape[ci][2];
            gc.d_ff    = 2 * gc.d_model;
            gc.context = 8;
            gc.vocab   = 261;
            char err[160];
            if (aster_config_validate(&gc, err, sizeof err) != 0) {
                aster_warn("gradient check: %s", err);
                check(0, "analytic gradients match a finite difference", &failures);
                continue;
            }

            AsterModel *gm = aster_model_new(&gc, 7u, 1);
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

    AsterConfig mcfg;
    aster_config_default(&mcfg);
    char err[512];
    puts("");
    if (aster_train(&tc, &mcfg, &ts, a.n_valid ? &vs : NULL, a.out, err, sizeof err) != 0) {
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

    aster_info("model %s: %d parameters, %d-token context, trained %u step(s)",
               file_name_of(a.model), m->off.total, m->cfg.context, step);
    int budget = aster_prompt_budget(&m->cfg, gp.max_new_tokens, (int)strlen(a.system ? a.system : ""));
    aster_info("sampling: %s, temperature %.2f, top_k %d, seed %u; at most %d prompt bytes fit",
               gp.temperature <= 0.0f ? "greedy" : "stochastic",
               (double)gp.temperature, gp.top_k, gp.seed, budget);
    if (a.prompt && strlen(a.prompt) > (size_t)budget)
        aster_warn("the prompt is longer than the %d bytes that fit, so its start will be dropped", budget);
    puts("");

    char *out = (char *)aster_xmalloc(m->cfg.context * 4 + 64);
    int hit_end = 0, truncated = 0;
    aster_generate(m, a.prompt ? a.prompt : "", &gp, out, m->cfg.context * 4 + 64, &hit_end, &truncated);
    puts("--- generated text (raw model output) ---");
    puts(out);
    puts("--- end ---");
    if (hit_end) aster_info("stopped on the END marker");
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

    char err[512];
    AsterDataset *d = dataset_build(&vs, err, sizeof err);
    if (!d) { aster_fail("%s", err); return 1; }

    double ppl = 0.0;
    double loss = dataset_eval_loss(m, d, 8, &ppl);
    if (loss < 0.0) { dataset_free(d); aster_fail("evaluation failed"); return 1; }

    aster_info("model:    %s (%d parameters, step %u, seed %u)",
               file_name_of(a.model), m->off.total, step, seed);
    aster_info("data:     %d window(s), %zu tokens, hash %s", d->n_win, d->tokens, d->hash);
    aster_info("held-out cross-entropy %.4f nats/token (perplexity %.2f)", loss, ppl);
    aster_info("Compare Aster versions on the SAME evaluation set before claiming improvement.");
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
    if (!strcmp(cmd, "generate")) return cmd_generate(argc, argv);
    if (!strcmp(cmd, "serve"))    return cmd_serve(argc, argv);
    if (!strcmp(cmd, "eval"))     return cmd_eval(argc, argv);

    fprintf(stderr, "error: unknown command \"%s\"\n\n", cmd);
    print_help();
    return 2;
}
