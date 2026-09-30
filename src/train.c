#include "train.h"
#include "jsonstr.h"
#include "util.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Ctrl+C asks for a clean stop: save what we have and exit 0. The previous
 * good checkpoint is never left half written, because checkpoints go through
 * aster_write_file_atomic(). */
static volatile sig_atomic_t g_stop = 0;
static void on_interrupt(int sig) { (void)sig; g_stop = 1; }

void train_config_default(TrainConfig *c) {
    c->batch = 8;
    c->steps = 1000;
    c->log_every = 25;
    c->warmup = 100;
    c->lr = 1e-3f;
    c->min_ratio = 0.1f;
    c->weight_decay = 0.01f;
    c->beta1 = 0.9f;
    c->beta2 = 0.95f;
    c->eps = 1e-8f;
    c->clip = 1.0f;
    c->seed = 1234u;
}

/* =============================================================== dataset */

/* A dataset's hash identifies the exact inputs a run used, so a log can be
 * tied to its data without re-hashing files by hand. Each file contributes
 * its path and its own SHA-256; the dataset hash is the SHA-256 of those
 * lines together. Hashing per file and then combining keeps this to one
 * pass per file instead of buffering the whole corpus.
 *
 * This used to be FNV-1a-32 formatted into a 65-byte field documented as a
 * SHA-256, which made the buffer size and the comment both wrong. */
typedef struct { char *p; size_t n, cap; } HashAcc;

static void hacc_add(HashAcc *h, const char *s, size_t n) {
    if (h->n + n + 1 > h->cap) {
        while (h->n + n + 1 > h->cap) h->cap = h->cap ? h->cap * 2 : 256;
        h->p = (char *)aster_xrealloc(h->p, h->cap);
    }
    memcpy(h->p + h->n, s, n);
    h->n += n;
}

static void hacc_add_file(HashAcc *h, const char *path, const char *data, size_t len) {
    char hex[65];
    aster_sha256_hex(data, len, hex);
    hacc_add(h, path, strlen(path));
    hacc_add(h, "\n", 1);
    hacc_add(h, hex, 64);
    hacc_add(h, "\n", 1);
}

static void hacc_finish(HashAcc *h, char out[65]) {
    aster_sha256_hex(h->p ? h->p : "", h->n, out);
    free(h->p);
    h->p = NULL; h->n = h->cap = 0;
}

/* Encodes one field and appends it with a single weight for every token it
 * produced. The weight is per POSITION, not per byte, so a five-byte word
 * that becomes one token contributes one weight, not five -- which is the
 * whole point of a sub-word tokenizer and the reason this cannot stay
 * byte-shaped. */
static int push_text(const AsterVocab *v, TokenList *tl, const char *text,
                     float weight, float *wbuf, int *nw, int *cap) {
    TokenList enc;
    token_list_init(&enc);
    const int rc = tok_encode(v, text, strlen(text), &enc);
    if (rc != 0) { token_list_free(&enc); return rc; }
    for (size_t i = 0; i < enc.count; ++i) {
        if (*nw >= *cap) { token_list_free(&enc); return -1; }
        token_list_push(tl, enc.ids[i]);
        wbuf[(*nw)++] = weight;
    }
    token_list_free(&enc);
    return 0;
}

/* ---------------------------------------------------------------- text mode
 * Concatenates the files with an END marker between them, then takes every
 * possible block-length window. The END marker stops the model from learning
 * that one file's last line and the next file's first line are one sentence. */

static AsterDataset *build_text(const DataSpec *spec, char *err, size_t errlen) {
    TokenList stream;
    token_list_init(&stream);
    size_t total_bytes = 0;
    HashAcc hacc = {NULL, 0, 0};

    for (int i = 0; i < spec->n_paths; ++i) {
        size_t len = 0; int ok = 0;
        char *txt = aster_read_file(spec->paths[i], &len, &ok);
        if (!ok) {
            snprintf(err, errlen, "cannot read '%s'", spec->paths[i]);
            token_list_free(&stream);
            return NULL;
        }
        aster_info("read  %-44s %9zu bytes", spec->paths[i], len);
        total_bytes += len;
        hacc_add_file(&hacc, spec->paths[i], txt, len);
        if (tok_encode(spec->v, txt, len, &stream) != 0) {
            snprintf(err, errlen, "cannot encode '%s' (it contains a NUL byte)", spec->paths[i]);
            free(txt);
            token_list_free(&stream);
            return NULL;
        }
        token_list_push(&stream, TOK_END);
        free(txt);
    }

    AsterDataset *d = (AsterDataset *)aster_xcalloc(1, sizeof *d);
    d->stream = stream.ids;
    d->stream_len = stream.count;
    d->tokens = stream.count;
    d->source_bytes = total_bytes;
    d->docs = spec->n_paths;
    d->block = spec->block;
    hacc_finish(&hacc, d->hash);

    if (d->stream_len < (size_t)spec->block + 1) {
        snprintf(err, errlen,
                 "pretraining text is too short: %zu tokens for a %d-token block (need %d)",
                 d->stream_len, spec->block, spec->block + 1);
        dataset_free(d);
        return NULL;
    }

    int n = (int)(d->stream_len - (size_t)spec->block);
    if (spec->max_examples > 0 && n > spec->max_examples) n = spec->max_examples;
    d->win = (AsterWindow *)aster_xcalloc((size_t)n, sizeof *d->win);
    d->n_win = n;
    d->ones = (float *)aster_xmalloc((size_t)spec->block * sizeof(float));
    for (int i = 0; i < spec->block; ++i) d->ones[i] = 1.0f;
    for (int i = 0; i < n; ++i) {
        d->win[i].start = (size_t)i;
        d->win[i].len = spec->block;
        d->win[i].w = d->ones;
    }
    return d;
}

/* --------------------------------------------------------------- chat mode
 * One JSON object per line: {"role": ..., "content": ...}.
 * A conversation must start with a system turn and then alternate strictly
 * user, assistant, user, assistant. Each complete pair becomes one training
 * window with the loss masked off everywhere except the assistant reply and
 * the END marker that closes it, so the model is never trained to predict the
 * question it was just given. */

typedef struct {
    char *system;
    char *user;
    char *assistant;
    int   expect_user;
} ChatConv;

static void conv_reset(ChatConv *c) {
    free(c->system); free(c->user); free(c->assistant);
    memset(c, 0, sizeof *c);
    c->expect_user = 1;
}

typedef struct { uint16_t *ids; int len; float *w; } RawWindow;

static AsterDataset *build_chat(const DataSpec *spec, char *err, size_t errlen) {
    RawWindow *raw = NULL;
    int n_raw = 0, cap_raw = 0, duplicates = 0;
    size_t total_bytes = 0;
    HashAcc hacc = {NULL, 0, 0};
    uint32_t *seen = NULL;
    size_t *seen_len = NULL;
    int n_seen = 0, cap_seen = 0;
    int conv_count = 0;

    for (int fi = 0; fi < spec->n_paths && err[0] == '\0'; ++fi) {
        size_t flen = 0; int ok = 0;
        char *text = aster_read_file(spec->paths[fi], &flen, &ok);
        if (!ok) { snprintf(err, errlen, "cannot read '%s'", spec->paths[fi]); goto fail; }
        aster_info("read  %-44s %9zu bytes", spec->paths[fi], flen);
        total_bytes += flen;
        hacc_add_file(&hacc, spec->paths[fi], text, flen);

        ChatConv c;
        memset(&c, 0, sizeof c);
        c.expect_user = 1;
        int have_system = 0, line_no = 0;
        char *p = text, *end = text + flen;

        while (p < end) {
            char *nl = (char *)memchr(p, '\n', (size_t)(end - p));
            char *line = p;
            size_t llen = nl ? (size_t)(nl - p) : (size_t)(end - p);
            p = nl ? nl + 1 : end;
            while (llen && (line[llen - 1] == '\r' || line[llen - 1] == ' ')) --llen;
            if (llen == 0) continue;
            ++line_no;

            char jerr[160] = {0};
            char role[32];
            int rc = json_get_string(line, llen, "role", role, sizeof role, jerr, sizeof jerr);
            if (rc != 0) {
                snprintf(err, errlen, "%s line %d: %s", spec->paths[fi], line_no,
                         rc == -2 ? "no \"role\" field" : jerr);
                goto fail_conv;
            }
            /* Decoded output is at most 4 bytes per input byte, so this is
             * always large enough and no length probe is needed. */
            size_t need = llen * 4 + 8;
            char *content = (char *)aster_xmalloc(need);
            rc = json_get_string(line, llen, "content", content, need, jerr, sizeof jerr);
            if (rc != 0) {
                snprintf(err, errlen, "%s line %d: %s", spec->paths[fi], line_no,
                         rc == -2 ? "no \"content\" field" : jerr);
                free(content);
                goto fail_conv;
            }

            if (strcmp(role, "system") == 0) {
                free(c.system);
                free(c.user); free(c.assistant);
                c.user = c.assistant = NULL;
                c.system = content;
                c.expect_user = 1;
                have_system = 1;
                conv_count++;
                continue;
            }
            if (!have_system) {
                snprintf(err, errlen, "%s line %d: a conversation must open with a system turn",
                         spec->paths[fi], line_no);
                free(content);
                goto fail_conv;
            }
            if (strcmp(role, "user") == 0) {
                if (!c.expect_user) {
                    snprintf(err, errlen, "%s line %d: expected an assistant turn before another user turn",
                             spec->paths[fi], line_no);
                    free(content);
                    goto fail_conv;
                }
                free(c.user);
                c.user = content;
                c.expect_user = 0;
                continue;
            }
            if (strcmp(role, "assistant") == 0) {
                if (c.expect_user || !c.user) {
                    snprintf(err, errlen, "%s line %d: an assistant turn must follow a user turn",
                             spec->paths[fi], line_no);
                    free(content);
                    goto fail_conv;
                }
                free(c.assistant);
                c.assistant = content;
                c.expect_user = 1;

                /* Build the window: SYSTEM sys USER q END ASSISTANT a END.
                 *
                 * MASK ALIGNMENT: dataset_gather() sets yb[i] = stream[i+1], so
                 * w[i] gates the prediction of token i+1. The first answer
                 * token is the target sitting immediately AFTER the ASSISTANT
                 * marker, so the ASSISTANT position itself must carry weight
                 * 1.0 -- not the first answer token. Weighting only the answer
                 * tokens (the obvious reading) silently drops the opening
                 * word of every answer from the loss, and the model then
                 * never learns to start a reply. The closing END has no
                 * successor, so it carries 0.0.
                 *
                 * This matters MORE under a sub-word tokenizer, not less. The
                 * dropped unit is no longer a character but a whole leading
                 * word -- "I" or "No" -- so the failure would be a model that
                 * reliably starts every reply one word late.
                 *
                 * The capacity is an upper bound computed from byte lengths
                 * (a token never covers more bytes than the text has), and
                 * then asserted against the real count below. Over-allocating
                 * is harmless; under-allocating would corrupt the mask.
                 */
                TokenList tl;
                token_list_init(&tl);
                int cw = 5 + (int)strlen(c.system) + (int)strlen(c.user) + (int)strlen(c.assistant);
                float *w = (float *)aster_xmalloc((size_t)cw * sizeof(float));
                int nw = 0, prc = 0;
                token_list_push(&tl, TOK_SYSTEM); w[nw++] = 0.0f;
                prc |= push_text(spec->v, &tl, c.system, 0.0f, w, &nw, &cw);
                token_list_push(&tl, TOK_USER);   w[nw++] = 0.0f;
                prc |= push_text(spec->v, &tl, c.user, 0.0f, w, &nw, &cw);
                token_list_push(&tl, TOK_END);    w[nw++] = 0.0f;
                token_list_push(&tl, TOK_ASSISTANT); w[nw++] = 1.0f;
                prc |= push_text(spec->v, &tl, c.assistant, 1.0f, w, &nw, &cw);
                token_list_push(&tl, TOK_END);    w[nw++] = 0.0f;
                if (prc != 0 || nw != (int)tl.count) {
                    /* Either a NUL slipped through, or the weight vector and
                     * the token list have come apart -- which would silently
                     * misalign every weight after the first divergence. */
                    snprintf(err, errlen,
                             "%s: could not build a training window for a conversation "
                             "(%s)", spec->paths[fi],
                             prc != 0 ? "its text contains a NUL byte"
                                      : "the loss mask and the token stream disagree");
                    token_list_free(&tl);
                    free(w);
                    goto fail_conv;
                }

                uint32_t wh = 2166136261u;
                for (size_t z = 0; z < tl.count; ++z) wh = (wh ^ tl.ids[z]) * 16777619u;
                int dup = 0;
                for (int z = 0; z < n_seen; ++z)
                    if (seen[z] == wh && seen_len[z] == tl.count) { dup = 1; break; }
                if (dup) {
                    duplicates++;
                    token_list_free(&tl);
                    free(w);
                } else {
                    if (n_seen == cap_seen) {
                        cap_seen = cap_seen ? cap_seen * 2 : 64;
                        seen = (uint32_t *)aster_xrealloc(seen, (size_t)cap_seen * sizeof(uint32_t));
                        seen_len = (size_t *)aster_xrealloc(seen_len, (size_t)cap_seen * sizeof(size_t));
                    }
                    seen[n_seen] = wh;
                    seen_len[n_seen] = tl.count;
                    n_seen++;
                    if (n_raw == cap_raw) {
                        cap_raw = cap_raw ? cap_raw * 2 : 64;
                        raw = (RawWindow *)aster_xrealloc(raw, (size_t)cap_raw * sizeof *raw);
                    }
                    raw[n_raw].ids = tl.ids;
                    raw[n_raw].len = (int)tl.count;
                    raw[n_raw].w = w;
                    n_raw++;
                }
                continue;
            }

            snprintf(err, errlen, "%s line %d: role must be system, user, or assistant (got \"%s\")",
                     spec->paths[fi], line_no, role);
            free(content);
            goto fail_conv;

        fail_conv:
            conv_reset(&c);
            free(text);
            goto fail;
        }

        if (!have_system) {
            snprintf(err, errlen, "%s: no system turn was found", spec->paths[fi]);
            free(text);
            goto fail;
        }
        if (!c.expect_user) {
            snprintf(err, errlen,
                     "%s: the file ends with a user turn that has no assistant reply. "
                     "Every conversation must be complete, or training would teach the model "
                     "to produce questions instead of answers.", spec->paths[fi]);
            conv_reset(&c);
            free(text);
            goto fail;
        }
        conv_reset(&c);
        free(text);
    }

    if (n_raw == 0) {
        snprintf(err, errlen, "no complete user/assistant pairs were found");
        goto fail;
    }

    AsterDataset *d = (AsterDataset *)aster_xcalloc(1, sizeof *d);
    d->n_win = n_raw;
    d->block = spec->block;
    d->docs = conv_count;
    d->source_bytes = total_bytes;
    hacc_finish(&hacc, d->hash);

    size_t total = 0;
    for (int i = 0; i < n_raw; ++i) total += (size_t)raw[i].len;
    d->stream = (uint16_t *)aster_xmalloc(total * sizeof(uint16_t));
    d->wbuf = (float *)aster_xmalloc(total * sizeof(float));
    d->stream_len = d->tokens = total;
    d->win = (AsterWindow *)aster_xcalloc((size_t)n_raw, sizeof *d->win);

    size_t off = 0;
    for (int i = 0; i < n_raw; ++i) {
        const uint16_t *src = raw[i].ids;
        const float *wsrc = raw[i].w;
        int len = raw[i].len;
        if (len > spec->block) {
            /* Keep the tail: the reply and its END marker matter more than
             * the start of the system prompt. */
            src += len - spec->block;
            wsrc += len - spec->block;
            len = spec->block;
            d->truncated++;
        }
        memcpy(d->stream + off, src, (size_t)len * sizeof(uint16_t));
        memcpy(d->wbuf + off, wsrc, (size_t)len * sizeof(float));
        d->win[i].start = off;
        d->win[i].len = len;
        d->win[i].w = d->wbuf + off;
        off += (size_t)len;
        free(raw[i].ids);
        free(raw[i].w);
    }
    free(raw); free(seen); free(seen_len);
    if (duplicates) aster_info("dropped %d exactly duplicated conversation(s)", duplicates);
    return d;

fail:
    if (raw) {
        for (int i = 0; i < n_raw; ++i) { free(raw[i].ids); free(raw[i].w); }
        free(raw);
    }
    free(seen); free(seen_len);
    return NULL;
}

AsterDataset *dataset_build(const DataSpec *spec, char *err, size_t errlen) {
    if (spec->n_paths < 1) { snprintf(err, errlen, "no data files were given"); return NULL; }
    if (spec->block < 8) { snprintf(err, errlen, "block size must be at least 8 tokens"); return NULL; }
    err[0] = '\0';
    return (spec->mode == ASTER_DATA_CHAT) ? build_chat(spec, err, errlen)
                                          : build_text(spec, err, errlen);
}

void dataset_free(AsterDataset *d) {
    if (!d) return;
    free(d->stream);
    free(d->win);
    free(d->ones);
    free(d->wbuf);
    free(d);
}

/* Walks the same files build_chat does and hands out every text span the
 * model would be trained on.
 *
 * This exists so the vocabulary learner sees the same text the model does.
 * Learning from the raw file instead would spend merges on
 * {"role":"user","content":" -- punctuation the model never sees as text --
 * and quietly produce a vocabulary tuned to the wrong thing. The two would
 * still agree on the round trip, so nothing would look wrong; the merges
 * would just be worse.
 *
 * There is no validation parameter, on purpose. See the header. */
int dataset_scan_text(const DataSpec *spec, AsterTextFn fn, void *ud, char *err, size_t errlen) {
    if (spec->n_paths < 1) { snprintf(err, errlen, "no data files were given"); return -1; }
    if (spec->mode != ASTER_DATA_CHAT) {
        /* Text mode is one continuous stream, so the file is the text. */
        for (int i = 0; i < spec->n_paths; ++i) {
            size_t len = 0; int ok = 0;
            char *txt = aster_read_file(spec->paths[i], &len, &ok);
            if (!ok) { snprintf(err, errlen, "cannot read '%s'", spec->paths[i]); return -1; }
            fn(ud, txt, len);
            free(txt);
        }
        return 0;
    }

    for (int fi = 0; fi < spec->n_paths; ++fi) {
        size_t flen = 0; int ok = 0;
        char *text = aster_read_file(spec->paths[fi], &flen, &ok);
        if (!ok) { snprintf(err, errlen, "cannot read '%s'", spec->paths[fi]); return -1; }

        char *p = text, *end = text + flen;
        int have_system = 0, line_no = 0;
        while (p < end) {
            char *nl = (char *)memchr(p, '\n', (size_t)(end - p));
            char *line = p;
            size_t llen = nl ? (size_t)(nl - p) : (size_t)(end - p);
            p = nl ? nl + 1 : end;
            while (llen && (line[llen - 1] == '\r' || line[llen - 1] == ' ')) --llen;
            if (llen == 0) continue;
            ++line_no;

            char jerr[160] = {0};
            char role[32];
            if (json_get_string(line, llen, "role", role, sizeof role, jerr, sizeof jerr) != 0) {
                snprintf(err, errlen, "%s line %d: bad \"role\"", spec->paths[fi], line_no);
                free(text);
                return -1;
            }
            size_t need = llen * 4 + 8;
            char *content = (char *)aster_xmalloc(need);
            if (json_get_string(line, llen, "content", content, need, jerr, sizeof jerr) != 0) {
                snprintf(err, errlen, "%s line %d: bad \"content\"", spec->paths[fi], line_no);
                free(content); free(text);
                return -1;
            }
            if (strcmp(role, "system") == 0) have_system = 1;
            if (have_system) fn(ud, content, strlen(content));
            free(content);
        }
        if (!have_system) {
            snprintf(err, errlen, "%s: no system turn was found", spec->paths[fi]);
            free(text);
            return -1;
        }
        free(text);
    }
    return 0;
}

/* Gathers B windows into contiguous [B][block] arrays. Positions past the end
 * of a window become TOK_PAD with weight 0, so padding cannot reach the loss. */
static void dataset_gather(AsterDataset *d, const int *idx, int B, int block,
                           uint16_t *x, uint16_t *y, float *w) {
    for (int b = 0; b < B; ++b) {
        const AsterWindow *e = &d->win[idx[b]];
        uint16_t *xb = x + (size_t)b * block;
        uint16_t *yb = y + (size_t)b * block;
        float *wb = w + (size_t)b * block;
        for (int i = 0; i < block; ++i) {
            if (i + 1 < e->len) {
                xb[i] = d->stream[e->start + i];
                yb[i] = d->stream[e->start + i + 1];
                wb[i] = e->w ? e->w[i] : 1.0f;
            } else {
                xb[i] = TOK_PAD;
                yb[i] = TOK_PAD;
                wb[i] = 0.0f;
            }
        }
    }
}

/* Per-position cross-entropy, read back out of the logits the forward pass
 * already computed. aster_forward returns a weighted MEAN, which is enough to
 * report nats/token but not enough for nats/byte: byte-weighting needs each
 * position's own cross-entropy, because the two figures differ by exactly the
 * compression ratio only if the targets are weighted individually. */
static double row_cross_entropy(const float *logits, int V, int target) {
    float mx = -INFINITY;
    for (int i = 0; i < V; ++i) if (logits[i] > mx) mx = logits[i];
    if (!isfinite(mx)) return INFINITY;
    double sum = 0.0;
    for (int i = 0; i < V; ++i) sum += exp((double)logits[i] - (double)mx);
    return log(sum) + (double)mx - (double)logits[target];
}

double dataset_eval_loss_full(AsterModel *m, AsterDataset *d, int batch, AsterLoss *out) {
    const int block = d->block;
    const int V = m->cfg.vocab;
    int B = batch > 0 ? batch : 8;
    if (B > d->n_win) B = d->n_win;
    if (B < 1 || block > m->cfg.context) return -1.0;

    /* Byte length of every token id, so a target can be charged for the bytes
     * it actually covers. */
    uint16_t *tlen = (uint16_t *)aster_xmalloc((size_t)V * sizeof(uint16_t));
    for (int i = 0; i < V; ++i)
        tlen[i] = ((size_t)i < sizeof m->vocab.tok_len / sizeof m->vocab.tok_len[0])
                  ? m->vocab.tok_len[i] : 1;

    AsterActs *a = aster_acts_new(B, block, &m->cfg);
    uint16_t *x = (uint16_t *)aster_xmalloc((size_t)B * block * sizeof(uint16_t));
    uint16_t *y = (uint16_t *)aster_xmalloc((size_t)B * block * sizeof(uint16_t));
    float    *w = (float *)aster_xmalloc((size_t)B * block * sizeof(float));
    int *idx = (int *)aster_xmalloc((size_t)B * sizeof(int));

    double nats = 0.0, weight = 0.0, weight_byte = 0.0;
    for (int start = 0; start < d->n_win; start += B) {
        int n = d->n_win - start;
        if (n > B) n = B;
        for (int i = 0; i < B; ++i) idx[i] = start + (i < n ? i : n - 1);
        dataset_gather(d, idx, B, block, x, y, w);
        for (int i = n; i < B; ++i)          /* zero out the duplicated tail rows */
            memset(w + (size_t)i * block, 0, (size_t)block * sizeof(float));

        double wsum = 0.0;
        for (int i = 0; i < B * block; ++i) wsum += w[i];
        if (wsum <= 0.0) continue;
        if (aster_forward(m, a, x, y, w, NULL) < 0.0) {
            free(tlen); free(x); free(y); free(w); free(idx); aster_acts_free(a);
            return -1.0;
        }
        for (int i = 0; i < B * block; ++i) {
            if (w[i] <= 0.0f) continue;
            const int b = i / block, t = i % block;
            const float *lg = a->logits + ((size_t)b * block + t) * (size_t)V;
            const double ce = row_cross_entropy(lg, V, y[i]);
            if (!isfinite(ce)) {
                free(tlen); free(x); free(y); free(w); free(idx); aster_acts_free(a);
                return -1.0;
            }
            const double tb = (double)tlen[y[i]];
            nats       += (double)w[i] * ce;
            weight     += (double)w[i];
            weight_byte+= (double)w[i] * tb;
        }
    }
    free(tlen); free(x); free(y); free(w); free(idx);
    aster_acts_free(a);
    if (weight <= 0.0 || weight_byte <= 0.0) return -1.0;

    memset(out, 0, sizeof *out);
    out->nats_per_token = nats / weight;
    /* Total nats over total bytes -- the standard bits-per-byte.
     *
     * The numerator is the PLAIN sum of cross-entropies, not a byte-weighted
     * average of them. Each token's cross-entropy already accounts for the
     * uncertainty of every byte that token covers, so multiplying it by the
     * token's byte length a second time double-counts long tokens. Doing so
     * makes the number inflate with token length: on a held-out set here it
     * reported 5.26 nats/byte where the correct figure is 1.63, and the error
     * grew with the compression ratio, so it got worse exactly as the
     * tokenizer got better. */
    out->nats_per_byte  = nats / weight_byte;
    /* nats -> bits is a DIVISION by ln 2, not an exponentiation.
     *
     * exp() turns a log-probability into a probability, which is what
     * perplexity wants and what bits-per-byte does not want: a rate is not a
     * log-probability. exp(nats_per_byte / ln 2) is monotonically increasing,
     * so checkpoint *selection* was unaffected -- exp(a) < exp(b) exactly when
     * a < b -- but every printed number was wrong, and absurdly so: 1.63
     * nats/byte is 2.35 bits/byte, and the exponentiated form reported 10.48.
     *
     * Check it: log2(vocab) must come out the same whatever the tokenizer.
     * A random model over 1146 tokens is 10.16 bits per token; at 2.99 bytes
     * per token that is 3.40 bits per byte, not exp(3.40) = 30.0. */
    out->bits_per_byte  = out->nats_per_byte / log(2.0);
    out->token_ppl      = exp(out->nats_per_token);
    out->weight         = (size_t)weight;
    out->bytes          = (size_t)(weight_byte + 0.5);
    out->tokens         = (size_t)(weight + 0.5);
    return out->nats_per_token;
}

double dataset_eval_loss(AsterModel *m, AsterDataset *d, int batch, double *out_ppl) {
    AsterLoss L;
    const double r = dataset_eval_loss_full(m, d, batch, &L);
    if (r < 0.0) return -1.0;
    if (out_ppl) *out_ppl = L.token_ppl;
    return L.nats_per_token;
}

/* ============================================================== optimizer */

/* Decoupled weight decay is applied only to the square weight matrices. Norm
 * gains, biases, embeddings, and positions are excluded, which is the usual
 * AdamW convention and keeps the norm scales free to grow. */
static void build_decay_mask(const AsterConfig *cfg, const AsterOffsets *o, unsigned char *mask) {
    const int C = cfg->d_model, F = cfg->d_ff;
    memset(mask, 0, (size_t)o->total);
    for (int l = 0; l < cfg->n_layer; ++l) {
        int b = o->layer0 + l * o->per_layer;
        for (int k = 0; k < C * C; ++k) mask[b + o->lw_wq + k] = 1;
        for (int k = 0; k < C * C; ++k) mask[b + o->lw_wk + k] = 1;
        for (int k = 0; k < C * C; ++k) mask[b + o->lw_wv + k] = 1;
        for (int k = 0; k < C * C; ++k) mask[b + o->lw_wo + k] = 1;
        for (int k = 0; k < C * F; ++k) mask[b + o->lw_ff1 + k] = 1;
        for (int k = 0; k < F * C; ++k) mask[b + o->lw_ff2 + k] = 1;
    }
}

typedef struct {
    float *m, *v;
    float  b1, b2, eps, wd, clip;
    int    t;
} AdamW;

static void adam_clip(AsterModel *model, const AdamW *ad) {
    double sumsq = 0.0;
    for (int i = 0, n = model->off.total; i < n; ++i) {
        float g = model->grads[i];
        if (isfinite(g)) sumsq += (double)g * g;
    }
    if (!(sumsq > 0.0)) return;
    double norm = sqrt(sumsq);
    if (!isfinite(norm) || norm <= 0.0 || norm <= ad->clip) return;
    float scale = (float)(ad->clip / norm);
    for (int i = 0, n = model->off.total; i < n; ++i) {
        float g = model->grads[i];
        model->grads[i] = isfinite(g) ? g * scale : 0.0f;
    }
}

/* Linear warmup, then cosine decay down to min_ratio * lr. */
static float lr_at(const TrainConfig *cfg, int step) {
    if (cfg->warmup > 0 && step < cfg->warmup)
        return cfg->lr * (float)(step + 1) / (float)cfg->warmup;
    int span = cfg->steps - cfg->warmup;
    if (span < 1) span = 1;
    int t = step - cfg->warmup;
    if (t < 0) t = 0;
    if (t > span) t = span;
    float f = 0.5f * (1.0f + cosf((float)M_PI * (float)t / (float)span));
    return cfg->lr * (cfg->min_ratio + (1.0f - cfg->min_ratio) * f);
}

static void adam_step(AsterModel *model, AdamW *ad, const unsigned char *decay, float lr) {
    const int n = model->off.total;
    ad->t++;
    const float bc1 = 1.0f - powf(ad->b1, (float)ad->t);
    const float bc2 = 1.0f - powf(ad->b2, (float)ad->t);
    for (int i = 0; i < n; ++i) {
        float g = model->grads[i];
        if (!isfinite(g)) g = 0.0f;
        float m = ad->m[i] = ad->b1 * ad->m[i] + (1.0f - ad->b1) * g;
        float v = ad->v[i] = ad->b2 * ad->v[i] + (1.0f - ad->b2) * g * g;
        float upd = (m / bc1) / (sqrtf(v / bc2) + ad->eps);
        if (decay && decay[i]) upd += ad->wd * model->params[i];
        model->params[i] -= lr * upd;
        model->grads[i] = 0.0f;
    }
}

/* =============================================================== training */

int aster_train(const TrainConfig *tc, const AsterConfig *mcfg,
                const DataSpec *train_spec, const DataSpec *val_spec,
                const char *out_path, char *err, size_t errlen) {
    signal(SIGINT, on_interrupt);
    err[0] = '\0';

    aster_info("architecture: %d layer(s), d_model %d, %d head(s) of width %d, "
               "d_ff %d, context %d tokens, vocab %d",
               mcfg->n_layer, mcfg->d_model, mcfg->n_head,
               mcfg->d_model / mcfg->n_head, mcfg->d_ff, mcfg->context, mcfg->vocab);
    int nparams = aster_param_count(mcfg);
    aster_info("parameters:  %d (%.3f million), token embedding tied to the output head",
               nparams, nparams / 1e6);

    if (train_spec->block > mcfg->context) {
        snprintf(err, errlen, "block size %d exceeds the model context of %d tokens",
                 train_spec->block, mcfg->context);
        return -1;
    }
    if (val_spec && val_spec->n_paths > 0 && val_spec->block != train_spec->block) {
        snprintf(err, errlen, "validation block size (%d) must match the training block size (%d)",
                 val_spec->block, train_spec->block);
        return -1;
    }

    aster_info("building the training split");
    AsterDataset *tr = dataset_build(train_spec, err, errlen);
    if (!tr) return -1;
    aster_info("train split: %d window(s), %zu tokens, %zu source bytes, %d source(s)",
               tr->n_win, tr->tokens, tr->source_bytes, tr->docs);
    if (tr->truncated)
        aster_warn("%d window(s) were shortened to fit the %d-token block", tr->truncated, tr->block);
    aster_info("train hash:   %s", tr->hash);

    AsterDataset *va = NULL;
    if (val_spec && val_spec->n_paths > 0) {
        aster_info("building the validation split");
        va = dataset_build(val_spec, err, errlen);
        if (!va) { dataset_free(tr); return -1; }
        aster_info("valid split: %d window(s), %zu tokens, %zu source bytes, %d source(s)",
                   va->n_win, va->tokens, va->source_bytes, va->docs);
        aster_info("valid hash:   %s", va->hash);
    } else {
        aster_warn("no validation data was given, so no held-out loss can be reported");
    }

    AsterModel *model = aster_model_new(mcfg, train_spec->v, tc->seed, 1);
    const int B = tc->batch, block = tr->block;
    AsterActs *acts = aster_acts_new(B, block, mcfg);
    uint16_t *x = (uint16_t *)aster_xmalloc((size_t)B * block * sizeof(uint16_t));
    uint16_t *y = (uint16_t *)aster_xmalloc((size_t)B * block * sizeof(uint16_t));
    float    *w = (float *)aster_xmalloc((size_t)B * block * sizeof(float));
    int *order = (int *)aster_xmalloc((size_t)tr->n_win * sizeof(int));
    int *bidx  = (int *)aster_xmalloc((size_t)B * sizeof(int));
    for (int i = 0; i < tr->n_win; ++i) order[i] = i;

    uint32_t rng;
    aster_rng_seed_init(&rng, tc->seed ^ 0x9E3779B9u);
    for (int i = tr->n_win - 1; i > 0; --i) {
        int j = (int)(aster_rng_u32(&rng) % (uint32_t)(i + 1));
        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    AdamW ad;
    ad.m = (float *)aster_xcalloc((size_t)nparams, sizeof(float));
    ad.v = (float *)aster_xcalloc((size_t)nparams, sizeof(float));
    ad.b1 = tc->beta1; ad.b2 = tc->beta2; ad.eps = tc->eps;
    ad.wd = tc->weight_decay; ad.clip = tc->clip; ad.t = 0;
    unsigned char *decay = (unsigned char *)aster_xmalloc((size_t)nparams);
    build_decay_mask(mcfg, &model->off, decay);

    AsterLoss L = {0};
    double val0 = va ? dataset_eval_loss_full(model, va, B, &L) : 0.0;
    const double val0_bpb = L.bits_per_byte;   /* kept: best_val moves during the run */
    double tokens_seen = 0.0;
    if (va) {
        aster_info("step %6d  valid %.4f nats/token  %.4f bits/byte  (token ppl %8.2f)   (untrained)",
                   0, L.nats_per_token, L.bits_per_byte, L.token_ppl);
        /* An untrained model is uniform, so bits/byte must come out at
         * log2(vocab) divided by the bytes per scored token. Printing it makes
         * that a checkable identity rather than something to take on trust --
         * and a wrong byte count shows up here immediately. */
        aster_info("  untrained check: %.4f nats/token x %.2f bytes/token = %.4f nats/byte / ln2 = "
                   "%.4f bits/byte; log2(vocab=%d) / %.2f = %.4f  [%s]",
                   L.nats_per_token, (double)L.bytes / (double)(L.weight ? L.weight : 1),
                   L.nats_per_byte, L.bits_per_byte, mcfg->vocab,
                   (double)L.bytes / (double)(L.weight ? L.weight : 1),
                   log((double)mcfg->vocab) / log(2.0)
                       / ((double)L.bytes / (double)(L.weight ? L.weight : 1)),
                   fabs(L.bits_per_byte - log((double)mcfg->vocab) / log(2.0)
                       / ((double)L.bytes / (double)(L.weight ? L.weight : 1))) < 0.05
                       ? "consistent" : "INCONSISTENT -- byte accounting is wrong");
    }

    /* Keep the parameters from the step with the best held-out loss.
     *
     * A 124k-parameter model given a few hundred examples will drive its
     * training loss to zero and then keep going: memorisation, not learning.
     * The last step is therefore the worst place to stop, and reporting the
     * final loss would report overfitting rather than what the model achieved.
     * Selecting on held-out data is the standard fix, and it makes the
     * reported number the honest best this run reached. */
    float *best_params = NULL;
    /* Selection is on bits/byte, the same number the run reports. Selecting on
     * one metric and quoting another would mean reporting the nats/byte of a
     * checkpoint chosen to minimise something else. */
    double best_val = L.bits_per_byte;
    int best_step = 0;
    if (va) {
        best_params = (float *)aster_xmalloc((size_t)nparams * sizeof(float));
        memcpy(best_params, model->params, (size_t)nparams * sizeof(float));
    }

    aster_info("training: seed %u, %d step(s), batch %d x %d tokens, lr %.2e "
               "(warmup %d, cosine to %.0f%%), AdamW wd %.3f, clip %.2f",
               tc->seed, tc->steps, B, block, (double)tc->lr, tc->warmup,
               (double)tc->min_ratio * 100.0, (double)tc->weight_decay, (double)tc->clip);

    double ema = 0.0;
    const double t0 = aster_now_seconds();
    double last_t = t0, tokens_last = 0.0;
    int step = 0, cursor = 0, stopped_early = 0;
    char vbuf[48];

    for (step = 1; step <= tc->steps; ++step) {
        if (cursor + B > tr->n_win) {
            /* A fresh deterministic draw from the same seed, so a rerun
             * reproduces this exact ordering. */
            aster_rng_seed_init(&rng, (tc->seed ^ 0x9E3779B9u) + (uint32_t)step * 2654435761u);
            for (int i = tr->n_win - 1; i > 0; --i) {
                int j = (int)(aster_rng_u32(&rng) % (uint32_t)(i + 1));
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
            cursor = 0;
        }
        for (int i = 0; i < B; ++i) bidx[i] = order[cursor + i];
        cursor += B;
        dataset_gather(tr, bidx, B, block, x, y, w);

        double loss = aster_forward(model, acts, x, y, w, NULL);
        if (loss < 0.0 || !isfinite(loss)) {
            snprintf(err, errlen, "loss became non-finite at step %d. Try a lower --lr, "
                     "or check the data for binary content.", step);
            goto fail;
        }
        aster_backward(model, acts, x, y, w);
        adam_clip(model, &ad);
        adam_step(model, &ad, decay, lr_at(tc, step - 1));

        ema = (step == 1) ? loss : (0.95 * ema + 0.05 * loss);
        tokens_last += (double)B * block;
        tokens_seen  += (double)B * block;

        if (step % tc->log_every == 0 || step == tc->steps) {
            double now = aster_now_seconds();
            if (va) {
                dataset_eval_loss_full(model, va, B, &L);
                snprintf(vbuf, sizeof vbuf, "%.4f nats/tok  %.4f bits/byte", L.nats_per_token, L.bits_per_byte);
                if (L.bits_per_byte < best_val) {
                    best_val = L.bits_per_byte;
                    best_step = step;
                    memcpy(best_params, model->params, (size_t)nparams * sizeof(float));
                }
            } else {
                snprintf(vbuf, sizeof vbuf, "%s", "n/a");
            }
            double tps = (now - last_t) > 0.0 ? tokens_last / (now - last_t) : 0.0;
            aster_info("step %6d  train %.4f  valid %s  seen %8.0f tok  elapsed %6.1fs  "
                       "%6.0f tok/s  lr %.2e",
                       step, ema, vbuf, tokens_seen, now - t0, tps, (double)lr_at(tc, step - 1));
            last_t = now;
            tokens_last = 0.0;
        }
        if (g_stop) { stopped_early = 1; break; }
    }

    if (stopped_early) {
        aster_warn("stopped on request after %d step(s); saving what is trained so far", step - 1);
        --step;
    } else {
        aster_info("finished %d step(s) in %.1fs", tc->steps, aster_now_seconds() - t0);
    }

    double val1 = val0;
    if (va) {
        if (best_step > 0 && best_step < step) {
            aster_info("held-out loss was still rising at step %d; reverting to the best "
                       "held-out parameters from step %d", step - 1, best_step);
            memcpy(model->params, best_params, (size_t)nparams * sizeof(float));
            step = best_step;
        }
        val1 = dataset_eval_loss_full(model, va, B, &L);
        aster_info("held-out %.4f -> %.4f nats/token   %.4f -> %.4f bits/byte: %s",
                   val0, val1, val0_bpb, L.bits_per_byte,
                   L.bits_per_byte < val0_bpb ? "improved" : "NO IMPROVEMENT");
        aster_info("  nats/token is only comparable against a model with the SAME "
                   "vocabulary; bits/byte is the figure to compare across tokenizers.");
        /* The intermediate figure is printed so the headline can be checked by
         * hand instead of taken on trust: bits/byte is nats/byte divided by
         * ln 2, and nats/byte is nats/token times the bytes-per-token actually
         * observed on the held-out targets. */
        aster_info("  check: %.4f nats/token x %.2f bytes/token = %.4f nats/byte / ln2 = %.4f bits/byte "
                   "(over %zu scored tokens, %zu bytes)",
                   L.nats_per_token, (double)L.bytes / (double)(L.weight ? L.weight : 1),
                   L.nats_per_byte, L.bits_per_byte, L.weight, L.bytes);
        if (!(L.bits_per_byte < val0_bpb))
            aster_warn("the held-out loss did not fall. That is a real result and is "
                       "reported as such. More steps, a lower --lr, or more data may help.");
    }
    free(best_params);

    aster_checkpoint_save(model, out_path, ad.m, ad.v, (uint32_t)step, tc->seed,
                          (uint32_t)ASTER_DEFAULT_MAX_NEW, 0.0f);

    {   /* Sidecar metadata, so the recipe is readable without a decoder. */
        char meta[3072];
        int n = snprintf(meta, sizeof meta,
            "{\n"
            "  \"format\": \"aster-model-metadata\",\n"
            "  \"checkpoint_format\": %d,\n"
            "  \"arch_version\": %d,\n"
            "  \"tokenizer\": \"bpe-2\",\n"
            "  \"tokenizer_version\": %d,\n"
            "  \"merges\": %d,\n"
            "  \"bytes_per_token\": %.4f,\n"
            "  \"architecture\": { \"n_layer\": %d, \"n_head\": %d, \"d_model\": %d,\n"
            "    \"d_ff\": %d, \"context\": %d, \"vocab\": %d },\n"
            "  \"parameters\": %d,\n"
            "  \"weight_tying\": \"token embedding is also the output head\",\n"
            "  \"train_steps\": %d,\n"
            "  \"seed\": %u,\n"
            "  \"batch\": %d,\n"
            "  \"block_tokens\": %d,\n"
            "  \"optimizer\": \"AdamW\",\n"
            "  \"learning_rate\": %.8g,\n"
            "  \"warmup_steps\": %d,\n"
            "  \"weight_decay\": %.4g,\n"
            "  \"beta1\": %.4g,\n"
            "  \"beta2\": %.4g,\n"
            "  \"grad_clip\": %.4g,\n"
            "  \"train_data_hash\": \"%s\",\n"
            "  \"valid_data_hash\": \"%s\",\n"
            "  \"train_windows\": %d,\n"
            "  \"valid_windows\": %d,\n"
            "  \"valid_nats_per_token_start\": %.6f,\n"
            "  \"valid_nats_per_token_end\": %.6f,\n"
            "  \"valid_bits_per_byte_start\": %.6f,\n"
            "  \"valid_bits_per_byte_end\": %.6f,\n"
            "  \"tokens_seen\": %.0f,\n"
            "  \"checkpoint_selection\": \"best held-out bits-per-byte during training\",\n"
            "  \"best_step\": %d\n"
            "}\n",
            ASTER_CHECKPOINT_FORMAT, ASTER_ARCH_VERSION, ASTER_TOKENIZER_VERSION,
            mcfg->vocab - ASTER_BASE_VOCAB, model->vocab.bytes_per_token,
            mcfg->n_layer, mcfg->n_head, mcfg->d_model, mcfg->d_ff, mcfg->context,
            mcfg->vocab, nparams, step, tc->seed, B, block,
            (double)tc->lr, tc->warmup, (double)tc->weight_decay,
            (double)tc->beta1, (double)tc->beta2, (double)tc->clip,
            tr->hash, va ? va->hash : "", tr->n_win, va ? va->n_win : 0,
            val0, val1, val0_bpb, L.bits_per_byte, tokens_seen,
            best_step);
        if (n > 0 && (size_t)n < sizeof meta) {
            char path[1200];
            snprintf(path, sizeof path, "%s.meta.json", out_path);
            aster_write_file_atomic(path, meta, (size_t)n);
            aster_info("wrote %s", path);
        }
    }

    aster_info("wrote checkpoint %s", out_path);

    free(x); free(y); free(w); free(order); free(bidx);
    free(ad.m); free(ad.v); free(decay);
    aster_acts_free(acts);
    aster_model_free(model);
    dataset_free(va);
    dataset_free(tr);
    return 0;

fail:
    free(x); free(y); free(w); free(order); free(bidx);
    free(ad.m); free(ad.v); free(decay);
    aster_acts_free(acts);
    aster_model_free(model);
    dataset_free(va);
    dataset_free(tr);
    return -1;
}
