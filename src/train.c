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

static void push_bytes(TokenList *tl, const char *text, size_t len, float weight, float *wbuf, int *nw) {
    for (size_t i = 0; i < len; ++i) {
        if (text[i] == '\0') continue;
        token_list_push(tl, (uint16_t)(unsigned char)text[i]);
        if (wbuf) wbuf[(*nw)++] = weight;
    }
}

/* ---------------------------------------------------------------- text mode
 * Concatenates the files with an END marker between them, then takes every
 * possible block-length window. The END marker stops the model from learning
 * that one file's last line and the next file's first line are one sentence. */

static AsterDataset *build_text(const DataSpec *spec, char *err, size_t errlen) {
    TokenList stream;
    token_list_init(&stream);
    size_t total_bytes = 0;
    uint32_t hcrc = 2166136261u;

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
        for (const char *p = spec->paths[i]; *p; ++p) hcrc = (hcrc ^ (unsigned char)*p) * 16777619u;
        for (size_t k = 0; k < len; ++k) hcrc = (hcrc ^ (unsigned char)txt[k]) * 16777619u;
        push_bytes(&stream, txt, len, 1.0f, NULL, NULL);
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
    snprintf(d->hash, sizeof d->hash, "fnv1a-%08x", hcrc);

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
    uint32_t hcrc = 2166136261u;
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
        for (size_t k = 0; k < flen; ++k) hcrc = (hcrc ^ (unsigned char)text[k]) * 16777619u;

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
                 * w[i] gates the prediction of token i+1. The first answer byte
                 * is the target sitting immediately AFTER the ASSISTANT marker,
                 * so the ASSISTANT position itself must carry weight 1.0 -- not
                 * the first byte. Weighting only the answer bytes (the obvious
                 * reading) silently drops the first byte of every answer from
                 * the loss, and the model then never learns to open a reply.
                 * The closing END has no successor, so it carries 0.0.
                 */
                TokenList tl;
                token_list_init(&tl);
                int cw = 1 + (int)strlen(c.system) + 1
                             + (int)strlen(c.user) + 1
                             + 1 + (int)strlen(c.assistant) + 1;
                float *w = (float *)aster_xmalloc((size_t)cw * sizeof(float));
                int nw = 0;
                token_list_push(&tl, TOK_SYSTEM); w[nw++] = 0.0f;
                push_bytes(&tl, c.system, strlen(c.system), 0.0f, w, &nw);
                token_list_push(&tl, TOK_USER);   w[nw++] = 0.0f;
                push_bytes(&tl, c.user, strlen(c.user), 0.0f, w, &nw);
                token_list_push(&tl, TOK_END);    w[nw++] = 0.0f;
                token_list_push(&tl, TOK_ASSISTANT); w[nw++] = 1.0f;
                push_bytes(&tl, c.assistant, strlen(c.assistant), 1.0f, w, &nw);
                token_list_push(&tl, TOK_END);    w[nw++] = 0.0f;

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
    snprintf(d->hash, sizeof d->hash, "fnv1a-%08x", hcrc);

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

double dataset_eval_loss(AsterModel *m, AsterDataset *d, int batch, double *out_ppl) {
    const int block = d->block;
    int B = batch > 0 ? batch : 8;
    if (B > d->n_win) B = d->n_win;
    if (B < 1 || block > m->cfg.context) return -1.0;

    AsterActs *a = aster_acts_new(B, block, &m->cfg);
    uint16_t *x = (uint16_t *)aster_xmalloc((size_t)B * block * sizeof(uint16_t));
    uint16_t *y = (uint16_t *)aster_xmalloc((size_t)B * block * sizeof(uint16_t));
    float    *w = (float *)aster_xmalloc((size_t)B * block * sizeof(float));
    int *idx = (int *)aster_xmalloc((size_t)B * sizeof(int));

    double weighted = 0.0, weight = 0.0;
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
        double loss = 0.0;
        if (aster_forward(m, a, x, y, w, &loss) < 0.0 || !isfinite(loss)) {
            free(x); free(y); free(w); free(idx); aster_acts_free(a);
            return -1.0;
        }
        weighted += loss * wsum;
        weight += wsum;
    }
    free(x); free(y); free(w); free(idx);
    aster_acts_free(a);
    if (weight <= 0.0) return -1.0;
    double mean = weighted / weight;
    if (out_ppl) *out_ppl = exp(mean);
    return mean;
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

    AsterModel *model = aster_model_new(mcfg, tc->seed, 1);
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

    double ppl = 0.0;
    double val0 = va ? dataset_eval_loss(model, va, B, &ppl) : 0.0;
    if (va) aster_info("step %6d  valid loss %.4f  perplexity %8.2f   (untrained)", 0, val0, ppl);

    /* Keep the parameters from the step with the best held-out loss.
     *
     * A 124k-parameter model given a few hundred examples will drive its
     * training loss to zero and then keep going: memorisation, not learning.
     * The last step is therefore the worst place to stop, and reporting the
     * final loss would report overfitting rather than what the model achieved.
     * Selecting on held-out data is the standard fix, and it makes the
     * reported number the honest best this run reached. */
    float *best_params = NULL;
    double best_val = val0;
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

        if (step % tc->log_every == 0 || step == tc->steps) {
            double now = aster_now_seconds();
            if (va) {
                double v = dataset_eval_loss(model, va, B, &ppl);
                snprintf(vbuf, sizeof vbuf, "%.4f (ppl %7.2f)", v, ppl);
                if (v < best_val) {
                    best_val = v;
                    best_step = step;
                    memcpy(best_params, model->params, (size_t)nparams * sizeof(float));
                }
            } else {
                snprintf(vbuf, sizeof vbuf, "%s", "n/a");
            }
            double tps = (now - last_t) > 0.0 ? tokens_last / (now - last_t) : 0.0;
            aster_info("step %6d  train %.4f  valid %s  elapsed %6.1fs  %6.0f tok/s  lr %.2e",
                       step, ema, vbuf, now - t0, tps, (double)lr_at(tc, step - 1));
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
        val1 = dataset_eval_loss(model, va, B, &ppl);
        aster_info("held-out loss %.4f -> %.4f  (perplexity %.2f): %s",
                   val0, val1, ppl, val1 < val0 ? "improved" : "NO IMPROVEMENT");
        if (!(val1 < val0))
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
            "  \"tokenizer\": \"byte-v1\",\n"
            "  \"tokenizer_version\": %d,\n"
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
            "  \"valid_loss_start\": %.6f,\n"
            "  \"valid_loss_end\": %.6f,\n"
            "  \"checkpoint_selection\": \"best held-out loss during training\",\n"
            "  \"best_step\": %d\n"
            "}\n",
            ASTER_CHECKPOINT_FORMAT, ASTER_ARCH_VERSION, ASTER_TOKENIZER_VERSION,
            mcfg->n_layer, mcfg->n_head, mcfg->d_model, mcfg->d_ff, mcfg->context,
            mcfg->vocab, nparams, step, tc->seed, B, block,
            (double)tc->lr, tc->warmup, (double)tc->weight_decay,
            (double)tc->beta1, (double)tc->beta2, (double)tc->clip,
            tr->hash, va ? va->hash : "", tr->n_win, va ? va->n_win : 0, val0, val1,
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
