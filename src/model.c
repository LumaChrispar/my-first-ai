#include "model.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GELU_K 0.7978845608028654f   /* sqrt(2/pi) */

/* Index helpers; B, T, L, H are AsterActs fields. */
#define IDX(b, t, w)     (((size_t)(b) * (size_t)T + (size_t)(t)) * (size_t)(w))
#define IDXL(b, l, t, w) ((((size_t)(b) * (size_t)L + (size_t)(l)) * (size_t)T + (size_t)(t)) * (size_t)(w))
/* xin and dxin hold L+1 residual stages, so they need their own stride.
 * IDXL strides by L, which is right for the [B][L][...] backward scratch
 * but addresses the wrong block for the residual stream. */
#define IDXR(b, l, t, w) ((((size_t)(b) * ((size_t)L + 1) + (size_t)(l)) * (size_t)T + (size_t)(t)) * (size_t)(w))
#define IDXH(b, l, h, t) (((((size_t)(b) * (size_t)L + (size_t)(l)) * (size_t)H + (size_t)(h)) \
                            * (size_t)T + (size_t)(t)) * (size_t)T)

/* ================================================================ config */

void aster_config_default(AsterConfig *cfg) {
    cfg->n_layer = ASTER_DEFAULT_LAYERS;
    cfg->n_head  = ASTER_DEFAULT_HEADS;
    cfg->d_model = ASTER_DEFAULT_DMODEL;
    cfg->d_ff    = ASTER_DEFAULT_DFF;
    cfg->context = ASTER_DEFAULT_CTX;
    cfg->vocab   = ASTER_VOCAB;
}

int aster_config_validate(const AsterConfig *cfg, char *err, size_t errlen) {
#define BAD(msg) do { snprintf(err, errlen, "%s", (msg)); return -1; } while (0)
    if (cfg->n_layer < 1 || cfg->n_layer > 64)   BAD("n_layer must be 1..64");
    if (cfg->n_head  < 1 || cfg->n_head  > 64)   BAD("n_head must be 1..64");
    if (cfg->d_model < 8 || cfg->d_model > 4096) BAD("d_model must be 8..4096");
    if (cfg->d_ff    < 1 || cfg->d_ff > 65536)   BAD("d_ff must be 1..65536");
    if (cfg->context < 8 || cfg->context > 8192) BAD("context must be 8..8192");
    if (cfg->vocab != ASTER_VOCAB)               BAD("vocab is fixed at 261 for the byte tokenizer");
    if (cfg->d_model % cfg->n_head != 0)         BAD("d_model must be divisible by n_head");
    if ((long long)cfg->d_model * cfg->d_ff > 8000000LL) BAD("d_model * d_ff is unreasonably large");
    return 0;
#undef BAD
}

void aster_offsets_init(AsterOffsets *o, const AsterConfig *cfg) {
    const int C = cfg->d_model, F = cfg->d_ff, V = cfg->vocab, T = cfg->context;
    int p = 0;
    o->tok_emb = p; p += V * C;
    o->pos_emb = p; p += T * C;
    o->layer0  = p;
    /* Layer-local offsets. ff1 is the only projection that carries a bias,
     * so its block is C*F + F long and ff2 starts after that. */
    o->lw_ln1w = 0;                 o->lw_ln1b = C;
    o->lw_wq   = 2 * C;             o->lw_wk   = o->lw_wq + C * C;
    o->lw_wv   = o->lw_wk + C * C;  o->lw_wo   = o->lw_wv + C * C;
    o->lw_ln2w = o->lw_wo + C * C;  o->lw_ln2b = o->lw_ln2w + C;
    o->lw_ff1  = o->lw_ln2b + C;    o->lw_ff2  = o->lw_ff1 + C * F + F;
    o->per_layer = o->lw_ff2 + F * C;
    p += cfg->n_layer * o->per_layer;
    o->lnf_w = p; o->lnf_b = p + C;
    o->total = p + 2 * C;      /* the final norm's scale AND shift both count */
}

int aster_param_count(const AsterConfig *cfg) {
    AsterOffsets o;
    aster_offsets_init(&o, cfg);
    return o.total;
}

/* ============================================================== numerics */

/* y = xhat * w + b, where xhat is the zero-mean unit-variance projection of
 * x. Epsilon guards the division when a channel happens to be constant. */
static void layer_norm_fwd(const float *x, const float *w, const float *b,
                           float *y, int n) {
    double mu = 0.0, var = 0.0;
    for (int i = 0; i < n; ++i) mu += x[i];
    mu /= n;
    for (int i = 0; i < n; ++i) { double d = x[i] - mu; var += d * d; }
    var /= n;
    double inv = 1.0 / sqrt(var + (double)ASTER_LN_EPS);
    for (int i = 0; i < n; ++i) y[i] = (float)(((x[i] - mu) * inv) * w[i] + b[i]);
}

/* dx = w/sigma * (dy - mean(dy) - xhat * mean(dy * xhat))
 * dw += dy * xhat,  db += dy.  Pass NULL for dw/db when they are not needed
 * (they are only trained for the two layer norms that actually have them). */
static void layer_norm_bwd(const float *x, const float *dy, const float *w,
                           float *dx, int n, float *dw, float *db) {
    double mu = 0.0, var = 0.0, m1 = 0.0, m2 = 0.0;
    for (int i = 0; i < n; ++i) mu += x[i];
    mu /= n;
    for (int i = 0; i < n; ++i) { double d = x[i] - mu; var += d * d; }
    var /= n;
    double inv = 1.0 / sqrt(var + (double)ASTER_LN_EPS);
    for (int i = 0; i < n; ++i) { double xh = (x[i] - mu) * inv; m1 += dy[i]; m2 += dy[i] * xh; }
    m1 /= n; m2 /= n;
    for (int i = 0; i < n; ++i) {
        double xh = (x[i] - mu) * inv;
        dx[i] = (float)(w[i] * inv * (dy[i] - m1 - xh * m2));
        if (dw) dw[i] += dy[i] * (float)xh;
        if (db) db[i] += dy[i];
    }
}

static inline float gelu_f(float x) {
    return 0.5f * x * (1.0f + tanhf(GELU_K * (x + 0.044715f * x * x * x)));
}

static inline float gelu_d(float x) {
    float t = tanhf(GELU_K * (x + 0.044715f * x * x * x));
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * GELU_K * (1.0f + 3.0f * 0.044715f * x * x);
}

/* y[o] = sum_k x[k]*W[o*in+k] + bias[o]   (W is [out][in]) */
static void linear_fwd(const float *x, int in, const float *W, const float *bias,
                       float *y, int out) {
    for (int o = 0; o < out; ++o) {
        const float *w = W + (size_t)o * in;
        float s = bias ? bias[o] : 0.0f;
        for (int k = 0; k < in; ++k) s += x[k] * w[k];
        y[o] = s;
    }
}

/* dW[o][k] += dy[o]*x[k];  db[o] += dy[o] */
static void linear_grad_wb(const float *x, int in, const float *dy, int out,
                           float *dW, float *db) {
    for (int o = 0; o < out; ++o) {
        float g = dy[o];
        if (db) db[o] += g;
        float *w = dW + (size_t)o * in;
        for (int k = 0; k < in; ++k) w[k] += g * x[k];
    }
}

/* x2[k] += sum_o dy[o]*W[o][k]   (gradient flowing through a weight matrix)
 * x2 and dy must not alias; use a scratch buffer when they would. */
static void linear_dx(const float *dy, int out, const float *W, float *x2, int in) {
    for (int o = 0; o < out; ++o) {
        float g = dy[o];
        const float *w = W + (size_t)o * in;
        for (int k = 0; k < in; ++k) x2[k] += g * w[k];
    }
}

/* Numerically stable softmax: subtract the row max before exponentiating.
 * Non-finite entries are dropped, and a fully degenerate row falls back to
 * uniform rather than producing NaN. */
static void softmax_row(float *v, int n) {
    float mx = -INFINITY;
    int any = 0;
    for (int i = 0; i < n; ++i) if (isfinite(v[i]) && (!any || v[i] > mx)) { mx = v[i]; any = 1; }
    if (!any) { for (int i = 0; i < n; ++i) v[i] = 1.0f / n; return; }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(v[i])) { v[i] = 0.0f; continue; }
        v[i] = expf(v[i] - mx);
        sum += v[i];
    }
    if (!(sum > 0.0) || !isfinite(sum)) { for (int i = 0; i < n; ++i) v[i] = 1.0f / n; return; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

/* Stable log-sum-exp of a logit row. */
static float logsumexp(const float *v, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) if (isfinite(v[i]) && v[i] > mx) mx = v[i];
    if (!isfinite(mx)) mx = 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) if (isfinite(v[i])) sum += exp((double)(v[i] - mx));
    if (!(sum > 0.0)) return 0.0f;
    return mx + (float)log(sum);
}

/* ============================================================ allocation */

AsterModel *aster_model_new(const AsterConfig *cfg, uint32_t seed, int want_grads) {
    char err[160];
    if (aster_config_validate(cfg, err, sizeof err) != 0) aster_fail("%s", err);

    AsterModel *m = (AsterModel *)aster_xcalloc(1, sizeof *m);
    m->cfg = *cfg;
    aster_offsets_init(&m->off, cfg);
    m->head_dim = cfg->d_model / cfg->n_head;
    m->params = (float *)aster_xcalloc((size_t)m->off.total, sizeof(float));
    if (want_grads) m->grads = (float *)aster_xcalloc((size_t)m->off.total, sizeof(float));

    const AsterOffsets *o = &m->off;
    const int C = cfg->d_model, F = cfg->d_ff;
    float *P = m->params;
    uint32_t rng;
    aster_rng_seed_init(&rng, seed);

    const float emb_std = 0.02f;
    /* The two residual-branch outputs are scaled by 1/sqrt(2*n_layer) so the
     * residual stream variance does not grow with depth. */
    const float res_std = emb_std / sqrtf((float)(2 * cfg->n_layer));

    for (int i = 0; i < o->layer0; ++i) P[i] = emb_std * aster_rng_normal(&rng);

    for (int l = 0; l < cfg->n_layer; ++l) {
        int b = o->layer0 + l * o->per_layer;
        /* ln*_w and ln*_b are stored back to back as [C weights][C biases].
         * Testing i%2 here interleaves them, which leaves every odd channel
         * with gain 0 (normalization silently disabled) and every even channel
         * with bias 1. Split on i < C, not on parity. */
        for (int i = 0; i < 2 * C; ++i) P[b + o->lw_ln1w + i] = (i < C) ? 1.0f : 0.0f;
        for (int i = 0; i < 2 * C; ++i) P[b + o->lw_ln2w + i] = (i < C) ? 1.0f : 0.0f;
        for (int i = 0; i < C * C; ++i) P[b + o->lw_wq + i] = emb_std * aster_rng_normal(&rng);
        for (int i = 0; i < C * C; ++i) P[b + o->lw_wk + i] = emb_std * aster_rng_normal(&rng);
        for (int i = 0; i < C * C; ++i) P[b + o->lw_wv + i] = emb_std * aster_rng_normal(&rng);
        for (int i = 0; i < C * C; ++i) P[b + o->lw_wo + i] = res_std * aster_rng_normal(&rng);
        for (int i = 0; i < C * F; ++i) P[b + o->lw_ff1 + i] = emb_std * aster_rng_normal(&rng);
        for (int i = 0; i < F * C; ++i) P[b + o->lw_ff2 + i] = res_std * aster_rng_normal(&rng);
        /* ff1 keeps a bias; the other projections are bias-free in this
         * architecture, and their buffers simply do not exist. */
        for (int i = 0; i < F; ++i) P[b + o->lw_ff1 + C * F + i] = 0.0f;
    }
    for (int i = 0; i < 2 * C; ++i) P[o->lnf_w + i] = (i < C) ? 1.0f : 0.0f;
    return m;
}

void aster_model_free(AsterModel *m) {
    if (!m) return;
    free(m->params);
    free(m->grads);
    free(m);
}

AsterActs *aster_acts_new(int B, int T, const AsterConfig *cfg) {
    if (B < 1) B = 1;
    if (T < 1) T = 1;
    if (T > cfg->context) aster_fail("sequence length %d exceeds the %d-token context", T, cfg->context);

    const int L = cfg->n_layer, H = cfg->n_head, C = cfg->d_model, F = cfg->d_ff, V = cfg->vocab;
    AsterActs *a = (AsterActs *)aster_xcalloc(1, sizeof *a);
    a->B = B; a->T = T; a->L = L; a->H = H; a->C = C; a->F = F; a->V = V;

    size_t total = 0;
    total += (size_t)B * (L + 1) * T * C;   /* xin     */
    total += (size_t)B * L * T * C;         /* ln1_out */
    total += (size_t)B * L * T * 3 * C;     /* qkv     */
    total += (size_t)B * L * T * C;         /* att     */
    total += (size_t)B * L * H * T * T;     /* attp    */
    total += (size_t)B * L * T * C;         /* proj    */
    total += (size_t)B * L * T * C;         /* res1    */
    total += (size_t)B * L * T * C;         /* ln2_out */
    total += (size_t)B * L * T * F;         /* ffz     */
    total += (size_t)B * L * T * F;         /* ffh     */
    total += (size_t)B * L * T * C;         /* ffo     */
    total += (size_t)B * T * C;             /* lnf_out */
    total += (size_t)B * T * V;             /* logits  */
    total += (size_t)B * (L + 1) * T * C;   /* dxin    */
    total += (size_t)B * L * T * C;         /* dres    */
    total += (size_t)B * L * T * C;         /* dln1    */
    total += (size_t)B * L * T * C;         /* datt    */
    total += (size_t)B * L * T * C;         /* dproj   */
    total += (size_t)B * L * T * C;         /* dln2    */
    total += (size_t)B * L * T * F;         /* dffh    */
    total += (size_t)B * L * T * 3 * C;     /* dqkv    */
    total += (size_t)B * T * C;             /* dlnf    */
    total += (size_t)B * T * V;             /* dlogits */

    a->xin = (float *)aster_xcalloc(1, aster_size_mul(total, sizeof(float)));
    /* p must start AFTER xin. Starting it at a->xin hands xin's own storage
     * to the first TAKE, so ln1_out aliases the residual stream and every
     * later buffer is shifted onto it. Because IDXR(b,l,t,C) and IDXL(b,l,t,C)
     * coincide for b=0 and l <= n_layer-1, layer_norm_fwd then writes ln1_out
     * over the block input of the very layer that is reading it. */
    size_t xin_count = (size_t)B * (L + 1) * T * C;
    float *p = a->xin + xin_count;
    #define TAKE(field, count) do { a->field = p; p += (size_t)(count); } while (0)
    TAKE(ln1_out, (size_t)B * L * T * C);
    TAKE(qkv,     (size_t)B * L * T * 3 * C);
    TAKE(att,     (size_t)B * L * T * C);
    TAKE(attp,    (size_t)B * L * H * T * T);
    TAKE(proj,    (size_t)B * L * T * C);
    TAKE(res1,    (size_t)B * L * T * C);
    TAKE(ln2_out, (size_t)B * L * T * C);
    TAKE(ffz,     (size_t)B * L * T * F);
    TAKE(ffh,     (size_t)B * L * T * F);
    TAKE(ffo,     (size_t)B * L * T * C);
    TAKE(lnf_out, (size_t)B * T * C);
    TAKE(logits,  (size_t)B * T * V);
    TAKE(dxin,    (size_t)B * (L + 1) * T * C);
    TAKE(dres,    (size_t)B * L * T * C);
    TAKE(dln1,    (size_t)B * L * T * C);
    TAKE(datt,    (size_t)B * L * T * C);
    TAKE(dproj,   (size_t)B * L * T * C);
    TAKE(dln2,    (size_t)B * L * T * C);
    TAKE(dffh,    (size_t)B * L * T * F);
    TAKE(dqkv,    (size_t)B * L * T * 3 * C);
    TAKE(dlnf,    (size_t)B * T * C);
    TAKE(dlogits, (size_t)B * T * V);
    #undef TAKE
    if ((size_t)(p - a->xin) != total) aster_fail("internal: activation arena size mismatch");
    return a;
}

AsterActs *aster_acts_resize(AsterActs *a, int B, int T, const AsterConfig *cfg) {
    if (a && a->B == B && a->T == T && a->L == cfg->n_layer &&
        a->C == cfg->d_model && a->F == cfg->d_ff) return a;
    aster_acts_free(a);
    return aster_acts_new(B, T, cfg);
}

void aster_acts_free(AsterActs *a) {
    if (!a) return;
    free(a->xin);
    free(a);
}

/* ================================================================ forward */

double aster_forward(AsterModel *m, AsterActs *a, const uint16_t *x,
                     const uint16_t *targets, const float *weights, double *loss) {
    const int B = a->B, T = a->T, L = a->L, H = a->H, C = a->C, F = a->F, V = a->V;
    const int hd = m->head_dim;
    const float inv_scale = 1.0f / sqrtf((float)hd);
    const AsterOffsets *o = &m->off;
    const float *P = m->params;

    if (T > m->cfg.context) return -1.0;
    for (int i = 0, n = B * T; i < n; ++i) if (x[i] >= V) return -1.0;

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float *te = P + o->tok_emb + (size_t)x[b * T + t] * C;
            const float *pe = P + o->pos_emb + (size_t)t * C;
            float *dst = a->xin + IDXR(b, 0, t, C);
            for (int c = 0; c < C; ++c) dst[c] = te[c] + pe[c];
        }
    }

    for (int l = 0; l < L; ++l) {
        const int base = o->layer0 + l * o->per_layer;
        const float *ln1w = P + base + o->lw_ln1w, *ln1b = P + base + o->lw_ln1b;
        const float *ln2w = P + base + o->lw_ln2w, *ln2b = P + base + o->lw_ln2b;
        const float *Wq = P + base + o->lw_wq, *Wk = P + base + o->lw_wk;
        const float *Wv = P + base + o->lw_wv, *Wo = P + base + o->lw_wo;
        const float *W1 = P + base + o->lw_ff1, *b1 = P + base + o->lw_ff1 + C * F;
        const float *W2 = P + base + o->lw_ff2;

        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                float *h1 = a->ln1_out + IDXL(b, l, t, C);
                layer_norm_fwd(a->xin + IDXR(b, l, t, C), ln1w, ln1b, h1, C);
                float *qkv = a->qkv + IDXL(b, l, t, 3 * C);
                linear_fwd(h1, C, Wq, NULL, qkv, C);
                linear_fwd(h1, C, Wk, NULL, qkv + C, C);
                linear_fwd(h1, C, Wv, NULL, qkv + 2 * C, C);
            }
            for (int h = 0; h < H; ++h) {
                for (int t = 0; t < T; ++t) {
                    float *p = a->attp + IDXH(b, l, h, t);
                    const float *qt = a->qkv + IDXL(b, l, t, 3 * C) + h * hd;
                    for (int s = 0; s <= t; ++s) {
                        const float *ks = a->qkv + IDXL(b, l, s, 3 * C) + C + h * hd;
                        float acc = 0.0f;
                        for (int d = 0; d < hd; ++d) acc += qt[d] * ks[d];
                        p[s] = acc * inv_scale;
                    }
                    for (int s = t + 1; s < T; ++s) p[s] = -1e30f;   /* strictly causal */
                    softmax_row(p, T);
                    float *ot = a->att + IDXL(b, l, t, C) + h * hd;
                    for (int d = 0; d < hd; ++d) ot[d] = 0.0f;
                    for (int s = 0; s <= t; ++s) {
                        const float *vs = a->qkv + IDXL(b, l, s, 3 * C) + 2 * C + h * hd;
                        float w = p[s];
                        for (int d = 0; d < hd; ++d) ot[d] += w * vs[d];
                    }
                }
            }
            for (int t = 0; t < T; ++t) {
                const size_t ic = IDXL(b, l, t, C);
                const size_t ir = IDXR(b, l, t, C);
                float *pr = a->proj + ic;
                linear_fwd(a->att + ic, C, Wo, NULL, pr, C);
                float *r1 = a->res1 + ic;
                const float *xin = a->xin + ir;
                for (int c = 0; c < C; ++c) r1[c] = xin[c] + pr[c];
                float *h2 = a->ln2_out + ic;
                layer_norm_fwd(r1, ln2w, ln2b, h2, C);
                float *f = a->ffh + IDXL(b, l, t, F);
                float *z = a->ffz + IDXL(b, l, t, F);
                linear_fwd(h2, C, W1, b1, z, F);
                /* gelu' must be evaluated at the pre-activation z, so z is kept
                 * rather than overwritten in place like ffh. */
                for (int i = 0; i < F; ++i) f[i] = gelu_f(z[i]);
                float *fo = a->ffo + ic;
                linear_fwd(f, F, W2, NULL, fo, C);
                float *nxt = a->xin + IDXR(b, l + 1, t, C);
                for (int c = 0; c < C; ++c) nxt[c] = r1[c] + fo[c];
            }
        }
    }

    /* Final layer norm, then the tied head: logits[v] = lnf_out . tok_emb[v]. */
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const size_t ih = IDX(b, t, C);
            layer_norm_fwd(a->xin + IDXR(b, L, t, C), P + o->lnf_w, P + o->lnf_b,
                           a->lnf_out + ih, C);
            const float *h = a->lnf_out + ih;
            float *lg = a->logits + IDX(b, t, V);
            for (int v = 0; v < V; ++v) {
                const float *te = P + o->tok_emb + (size_t)v * C;
                float acc = 0.0f;
                for (int c = 0; c < C; ++c) acc += h[c] * te[c];
                lg[v] = acc;
            }
        }
    }

    if (!weights || !targets) { if (loss) *loss = 0.0; return 0.0; }

    double wsum = 0.0, total = 0.0;
    for (int i = 0, n = B * T; i < n; ++i) wsum += weights[i];
    if (wsum <= 0.0) { if (loss) *loss = 0.0; return 0.0; }
    for (int i = 0, n = B * T; i < n; ++i) {
        if (weights[i] <= 0.0f) continue;
        if (targets[i] >= V) return -1.0;
        const float *lg = a->logits + IDX(0, i, V);
        total += weights[i] * ((double)logsumexp(lg, V) - lg[targets[i]]);
    }
    double mean = total / wsum;
    if (!isfinite(mean)) mean = INFINITY;
    if (loss) *loss = mean;
    return mean;
}

/* =============================================================== backward */

double aster_backward(AsterModel *m, AsterActs *a, const uint16_t *x,
                      const uint16_t *targets, const float *weights) {
    const int B = a->B, T = a->T, L = a->L, H = a->H, C = a->C, F = a->F, V = a->V;
    const int hd = m->head_dim;
    const float inv_scale = 1.0f / sqrtf((float)hd);
    const AsterOffsets *o = &m->off;
    float *P = m->params, *G = m->grads;
    if (!G || !weights || !targets) return 0.0;

    double wsum = 0.0;
    for (int i = 0, n = B * T; i < n; ++i) wsum += weights[i];
    if (wsum <= 0.0) return 0.0;

    /* dlogits = (softmax(logits) - onehot(target)) * weight / sum(weight) */
    for (int i = 0, n = B * T; i < n; ++i) {
        float *dl = a->dlogits + IDX(0, i, V);
        if (weights[i] <= 0.0f) { memset(dl, 0, sizeof(float) * V); continue; }
        memcpy(dl, a->logits + IDX(0, i, V), sizeof(float) * V);
        softmax_row(dl, V);
        float f = weights[i] / (float)wsum;
        for (int v = 0; v < V; ++v) dl[v] *= f;
        if (targets[i] < V) dl[targets[i]] -= f;
    }

    /* Tied head: logits = lnf_out @ tok_emb^T, so dlogits reaches both sides. */
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const size_t ih = IDX(b, t, C);
            const float *dl = a->dlogits + IDX(b, t, V);
            const float *h = a->lnf_out + ih;
            float *dh = a->dlnf + ih;
            memset(dh, 0, sizeof(float) * C);
            for (int v = 0; v < V; ++v) {
                float g = dl[v];
                if (g == 0.0f) continue;
                float *te = G + o->tok_emb + (size_t)v * C;
                const float *pte = P + o->tok_emb + (size_t)v * C;
                for (int c = 0; c < C; ++c) { te[c] += g * h[c]; dh[c] += g * pte[c]; }
            }
        }
    }

    memset(a->dxin, 0, sizeof(float) * (size_t)B * (L + 1) * T * C);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const size_t ic = IDXR(b, L, t, C);
            layer_norm_bwd(a->xin + ic, a->dlnf + IDX(b, t, C), P + o->lnf_w,
                           a->dxin + ic, C, G + o->lnf_w, G + o->lnf_b);
        }
    }

    for (int l = L - 1; l >= 0; --l) {
        const int base = o->layer0 + l * o->per_layer;
        const float *ln1w = P + base + o->lw_ln1w;
        const float *ln2w = P + base + o->lw_ln2w;
        const float *Wq = P + base + o->lw_wq, *Wk = P + base + o->lw_wk;
        const float *Wv = P + base + o->lw_wv, *Wo = P + base + o->lw_wo;
        const float *W1 = P + base + o->lw_ff1, *W2 = P + base + o->lw_ff2;
        float *gLn1 = G + base + o->lw_ln1w;      /* [2C] w then b */
        float *gLn2 = G + base + o->lw_ln2w;      /* [2C] w then b */
        /* The four projections are bias-free and sit back to back, so each
         * gradient pointer comes straight from its own offset. Chaining them
         * off the previous matrix plus C assumed a bias slot that does not
         * exist and shifted Wk, Wv and Wo by one row each. */
        float *gWq = G + base + o->lw_wq;
        float *gWk = G + base + o->lw_wk;
        float *gWv = G + base + o->lw_wv;
        float *gWo = G + base + o->lw_wo;
        float *gW1 = G + base + o->lw_ff1, *gb1 = gW1 + C * F;
        float *gW2 = G + base + o->lw_ff2;

        for (int b = 0; b < B; ++b) {
            /* ================= pass A: feed-forward, residual, d(att) =========
             * d(att) comes from the output projection, which is downstream of
             * the feed-forward, so this pass has to run BEFORE the attention
             * backward below. The ln1 path is deferred to pass B, which is
             * where the attention gradients become available. */
            for (int t = 0; t < T; ++t) {
                const size_t ic = IDXL(b, l, t, C);
                const size_t iff = IDXL(b, l, t, F);
                /* d(xin[l+1]) arrives in dxin[l+1]; this layer's result is
                 * d(xin[l]), so it is built in its own slot and stored back
                 * into dxin[l]. Accumulating straight into dxin[l+1] would
                 * overwrite the value the next layer down reads. */
                float *dtop = a->dres + IDXL(b, l, t, C);
                memcpy(dtop, a->dxin + IDXR(b, l + 1, t, C), sizeof(float) * (size_t)C);
                float *dffh = a->dffh  + iff;
                float *dln2 = a->dln2  + ic;
                float *dprj = a->dproj + ic;
                float *datt = a->datt  + ic;

                /* Feed-forward. xin[l+1] = res1 + ffo, so
                 *     d(ffo) = dtop                     (residual identity)
                 *     d(ffh) = d(ffo) @ W2^T
                 * Nothing is added to dtop for the ffo path: ffo is ADDED to
                 * res1, it is not composed with it. W2 is stored
                 * [d_model][d_ff], and the d(ffh) read needs the transposed
                 * view, which is exactly what linear_dx computes. */
                memset(dffh, 0, sizeof(float) * (size_t)F);
                linear_dx(dtop, C, W2, dffh, F);            /* d(ffh) = d(ffo) @ W2^T */
                linear_grad_wb(a->ffh + iff, F, dtop, C, gW2, NULL);

                /* dffh is d(ffz), indexed by the feed-forward's OUTPUT channel,
                 * so it has F entries. d(ln2_out) is indexed by d_model and has
                 * C entries: contracting through W1 is what changes length,
                 * and skipping it would feed F values into a C-wide consumer. */
                for (int f = 0; f < F; ++f) dffh[f] *= gelu_d(a->ffz[iff + f]);
                linear_grad_wb(a->ln2_out + ic, C, dffh, F, gW1, gb1);
                memset(dln2, 0, sizeof(float) * (size_t)C);
                linear_dx(dffh, F, W1, dln2, C);            /* d(ln2_out) */

                /* res1 is the ln2 input, and d(res1) splits evenly to xin[l]
                 * (identity path) and to proj. `dy` here must be d(ln2_out):
                 * the gradient with respect to the layer norm's OUTPUT.
                 *
                 * xin[l+1] = res1 + ffo, so res1 reaches the loss twice: once
                 * through the direct residual add and once through the feed
                 * forward. Only the second is the ln2 Jacobian, so dtop has to
                 * be added back on -- res1 enters the residual stream with
                 * coefficient 1. Dropping it silently starves the whole
                 * residual stream of the dominant part of its gradient. */
                layer_norm_bwd(a->res1 + ic, dln2, ln2w, dprj, C, gLn2, gLn2 + C);
                for (int c = 0; c < C; ++c) dprj[c] += dtop[c];   /* residual add */
                memcpy(dtop, dprj, sizeof(float) * (size_t)C);   /* d(xin[l]) */

                /* proj = att @ Wo^T with Wo stored [d_model][d_model], so
                 * d(att)[k] = sum_o d(proj)[o]*Wo[o][k] -- which is linear_dx.
                 * The transposed variant is the FORWARD product Wo @ d(proj)
                 * and produces a transposed gradient, which is why this
                 * mattered even though Wo is square here. */
                linear_grad_wb(a->att + ic, C, dprj, C, gWo, NULL);
                memset(datt, 0, sizeof(float) * (size_t)C);
                linear_dx(dprj, C, Wo, datt, C);

                /* Park the pass-A contribution; pass B adds the ln1 path. */
                memcpy(a->dxin + IDXR(b, l, t, C), dtop, sizeof(float) * (size_t)C);
            }

            /* ================= attention backward, head by head ==============
             * dqkv holds dq in [0,C), dk in [C,2C), dv in [2C,3C) per position.
             * dk and dv accumulate over every query t, dq belongs to its own t. */
            for (int s = 0; s < T; ++s) {
                for (int h = 0; h < H; ++h) {
                    float *dst = a->dqkv + IDXL(b, l, s, 3 * C);
                    for (int d = 0; d < hd; ++d) {
                        dst[C + h * hd + d] = 0.0f;            /* dk */
                        dst[2 * C + h * hd + d] = 0.0f;        /* dv */
                    }
                }
            }
            for (int t = 0; t < T; ++t) {
                for (int h = 0; h < H; ++h) {
                    float *dst = a->dqkv + IDXL(b, l, t, 3 * C);
                    for (int d = 0; d < hd; ++d) dst[h * hd + d] = 0.0f;   /* dq */
                }
            }
            for (int h = 0; h < H; ++h) {
                for (int t = 0; t < T; ++t) {
                    const float *p = a->attp + IDXH(b, l, h, t);
                    const float *dat = a->datt + IDXL(b, l, t, C) + h * hd;
                    /* Softmax backward.
                     *   score[s] = inv_scale * (q[t] . k[s])
                     *   p[s]     = softmax(score)[s]
                     *   att[t]   = sum_s p[s] * v[s]
                     * so dL/dscore[s] = p[s] * (dL/da[s] - sum_r p[r]*dL/da[r])
                     * with dL/da[s] = dat . v[s]. The softmax Jacobian's
                     * `rowsum` MUST be built from those same ds values, and
                     * inv_scale does NOT belong in ds -- it belongs on the
                     * q.k path, so it is applied once when the score gradient
                     * is spread back onto q and k. */
                    double rowsum = 0.0;
                    for (int s = 0; s <= t; ++s) {
                        const float *vs = a->qkv + IDXL(b, l, s, 3 * C) + 2 * C + h * hd;
                        float acc = 0.0f;
                        for (int d = 0; d < hd; ++d) acc += dat[d] * vs[d];
                        rowsum += (double)p[s] * (double)acc;
                    }
                    const float *qt = a->qkv + IDXL(b, l, t, 3 * C) + h * hd;
                    float *dqt = a->dqkv + IDXL(b, l, t, 3 * C) + h * hd;
                    for (int s = 0; s <= t; ++s) {
                        const float *ks = a->qkv + IDXL(b, l, s, 3 * C) + C + h * hd;
                        const float *vs = a->qkv + IDXL(b, l, s, 3 * C) + 2 * C + h * hd;
                        float *dks = a->dqkv + IDXL(b, l, s, 3 * C) + C + h * hd;
                        float *dvs = a->dqkv + IDXL(b, l, s, 3 * C) + 2 * C + h * hd;
                        float acc = 0.0f;
                        for (int d = 0; d < hd; ++d) acc += dat[d] * vs[d];
                        float ds = p[s] * (float)((double)acc - rowsum);
                        for (int d = 0; d < hd; ++d) {
                            dqt[d] += inv_scale * ds * ks[d];
                            dks[d] += inv_scale * ds * qt[d];
                            dvs[d] += p[s] * dat[d];
                        }
                    }
                }
            }

            /* ================= pass B: Q/K/V weights and the ln1 path ======== */
            for (int t = 0; t < T; ++t) {
                const size_t ic = IDXL(b, l, t, C);
                const size_t ir = IDXR(b, l, t, C);
                float *dln1 = a->dln1 + ic;
                float *dln2 = a->dln2 + ic;
                const float *dqkv = a->dqkv + IDXL(b, l, t, 3 * C);

                /* Q/K/V share one normalised input, so d(ln1_out) accumulates
                 * into dln1. It reaches xin[l] through the ln1 norm below, so
                 * d(xin[l]) is not touched here. */
                memset(dln1, 0, sizeof(float) * (size_t)C);
                linear_grad_wb(a->ln1_out + ic, C, dqkv, C, gWq, NULL);
                linear_dx(dqkv, C, Wq, dln1, C);
                linear_grad_wb(a->ln1_out + ic, C, dqkv + C, C, gWk, NULL);
                linear_dx(dqkv + C, C, Wk, dln1, C);
                linear_grad_wb(a->ln1_out + ic, C, dqkv + 2 * C, C, gWv, NULL);
                linear_dx(dqkv + 2 * C, C, Wv, dln1, C);

                /* ln1: add its gradient on top of the pass-A contribution. */
                layer_norm_bwd(a->xin + ir, dln1, ln1w, dln2, C, gLn1, gLn1 + C);
                float *dxl = a->dxin + IDXR(b, l, t, C);
                for (int c = 0; c < C; ++c) dxl[c] += dln2[c];
            }
        }
    }

    /* Token and position embedding gradients. */
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const size_t ir = IDXR(b, 0, t, C);
            float *pe = G + o->pos_emb + (size_t)t * C;
            float *te = G + o->tok_emb + (size_t)x[b * T + t] * C;
            for (int c = 0; c < C; ++c) { pe[c] += a->dxin[ir + c]; te[c] += a->dxin[ir + c]; }
        }
    }
    return 0.0;
}

/* ============================================================= checkpoint */

static void put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xFFu);
    p[1] = (unsigned char)((v >> 8) & 0xFFu);
    p[2] = (unsigned char)((v >> 16) & 0xFFu);
    p[3] = (unsigned char)((v >> 24) & 0xFFu);
}

static uint32_t get_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_f32(unsigned char *p, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    put_u32(p, bits);
}

static float get_f32(const unsigned char *p) {
    uint32_t bits = get_u32(p);
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* 14 u32 fields plus the f32 temperature_default slot. */
#define CK_HEADER_U32 15
#define CK_HEADER_BYTES (8 + 4 * CK_HEADER_U32)

void aster_checkpoint_save(const AsterModel *m, const char *path,
                           const float *adam_m, const float *adam_v,
                           uint32_t step, uint32_t seed,
                           uint32_t max_new_tokens, float temperature) {
    const AsterOffsets *o = &m->off;
    const uint32_t total = (uint32_t)o->total;
    const int has_opt = (adam_m && adam_v) ? 1 : 0;
    const size_t floats = (size_t)total * (has_opt ? 3u : 1u);
    const size_t bytes = CK_HEADER_BYTES + floats * 4;
    unsigned char *buf = (unsigned char *)aster_xmalloc(bytes);

    memcpy(buf, ASTER_CHECKPOINT_MAGIC, 8);
    size_t off = 8;
#define WU(v) do { put_u32(buf + off, (uint32_t)(v)); off += 4; } while (0)
#define WF(v) do { put_f32(buf + off, (float)(v)); off += 4; } while (0)
    WU(ASTER_CHECKPOINT_FORMAT);
    WU(ASTER_ARCH_VERSION);
    WU(ASTER_TOKENIZER_VERSION);
    WU(m->cfg.n_layer); WU(m->cfg.n_head); WU(m->cfg.d_model);
    WU(m->cfg.d_ff);    WU(m->cfg.context); WU(m->cfg.vocab);
    WU(total);
    WU(step); WU(seed); WU(has_opt); WU(max_new_tokens);
    WF(temperature);
#undef WU
#undef WF
    for (uint32_t i = 0; i < total; ++i) { put_f32(buf + off, m->params[i]); off += 4; }
    if (has_opt) {
        for (uint32_t i = 0; i < total; ++i) { put_f32(buf + off, adam_m[i]); off += 4; }
        for (uint32_t i = 0; i < total; ++i) { put_f32(buf + off, adam_v[i]); off += 4; }
    }
    if (off != bytes) aster_fail("internal: checkpoint size mismatch (%zu vs %zu)", off, bytes);
    aster_write_file_atomic(path, buf, bytes);
    free(buf);
}

AsterModel *aster_checkpoint_load(const char *path,
                                  float **out_m, float **out_v,
                                  uint32_t *out_step, uint32_t *out_seed,
                                  char *err, size_t errlen) {
#define BAD(msg) do { snprintf(err, errlen, "%s", (msg)); free(hdr); return NULL; } while (0)
    if (out_m) *out_m = NULL;
    if (out_v) *out_v = NULL;

    size_t flen = 0; int ok = 0;
    unsigned char *hdr = (unsigned char *)aster_read_file(path, &flen, &ok);
    if (!ok || !hdr) { snprintf(err, errlen, "cannot open checkpoint '%s'", path); return NULL; }

    if (flen < CK_HEADER_BYTES) BAD("file is truncated: too short to be a checkpoint");
    if (memcmp(hdr, ASTER_CHECKPOINT_MAGIC, 8) != 0) BAD("bad magic bytes: this is not an Aster checkpoint");

    size_t o = 8;
    uint32_t format = get_u32(hdr + o); o += 4;
    uint32_t arch   = get_u32(hdr + o); o += 4;
    uint32_t tokver = get_u32(hdr + o); o += 4;
    AsterConfig cfg;
    cfg.n_layer = (int)get_u32(hdr + o); o += 4;
    cfg.n_head  = (int)get_u32(hdr + o); o += 4;
    cfg.d_model = (int)get_u32(hdr + o); o += 4;
    cfg.d_ff    = (int)get_u32(hdr + o); o += 4;
    cfg.context = (int)get_u32(hdr + o); o += 4;
    cfg.vocab   = (int)get_u32(hdr + o); o += 4;
    uint32_t declared = get_u32(hdr + o); o += 4;
    uint32_t step = get_u32(hdr + o); o += 4;
    uint32_t seed = get_u32(hdr + o); o += 4;
    uint32_t has_opt = get_u32(hdr + o); o += 4;
    o += 4;  /* max_new_tokens */
    o += 4;  /* temperature */

    if (format != ASTER_CHECKPOINT_FORMAT) BAD("unsupported checkpoint format version");
    if (arch != ASTER_ARCH_VERSION)        BAD("checkpoint architecture version does not match this build");
    if (tokver != ASTER_TOKENIZER_VERSION) BAD("checkpoint tokenizer version does not match this build");
    if (aster_config_validate(&cfg, err, errlen) != 0) { free(hdr); return NULL; }
    if (declared != (uint32_t)aster_param_count(&cfg))
        BAD("checkpoint parameter count disagrees with its own configuration");
    if (has_opt > 1) BAD("checkpoint optimizer flag is corrupt");

    size_t want_floats = (size_t)declared * (has_opt ? 3u : 1u);
    if (flen != CK_HEADER_BYTES + want_floats * 4)
        BAD("checkpoint size does not match its declared contents");

    AsterModel *m = aster_model_new(&cfg, 1u, 0);
    for (uint32_t i = 0; i < declared; ++i) {
        m->params[i] = get_f32(hdr + o);
        o += 4;
        if (!isfinite(m->params[i])) {
            free(hdr); aster_model_free(m);
            snprintf(err, errlen, "checkpoint holds a non-finite weight at index %u", i);
            return NULL;
        }
    }
    if (has_opt) {
        size_t nb = (size_t)declared * sizeof(float);
        float *mm = (float *)aster_xmalloc(nb);
        float *vv = (float *)aster_xmalloc(nb);
        for (uint32_t i = 0; i < declared; ++i) { mm[i] = get_f32(hdr + o); o += 4; }
        for (uint32_t i = 0; i < declared; ++i) { vv[i] = get_f32(hdr + o); o += 4; }
        if (out_m) *out_m = mm; else free(mm);
        if (out_v) *out_v = vv; else free(vv);
    }
    free(hdr);
    if (out_step) *out_step = step;
    if (out_seed)  *out_seed  = seed;
    return m;
#undef BAD
}

/* ================================================================ sampling */

void gen_params_default(GenParams *g) {
    g->max_new_tokens = ASTER_DEFAULT_MAX_NEW;
    g->temperature = 0.0f;   /* greedy by default, so debugging is repeatable */
    g->top_k = 0;
    g->seed = 0;
    g->system = NULL;
}

int aster_prompt_budget(const AsterConfig *cfg, int max_new_tokens, int system_bytes) {
    int reserve = max_new_tokens;
    if (reserve > cfg->context - 1) reserve = cfg->context - 1;
    if (reserve < 1) reserve = 1;
    if (system_bytes < 0) system_bytes = 0;
    /* Fixed overhead is SYSTEM, USER, END, ASSISTANT; the system text sits
     * between SYSTEM and USER and is charged to the user's budget so a long
     * system prompt cannot silently overflow the context. */
    int budget = cfg->context - reserve - 4 - system_bytes;
    return budget < 0 ? 0 : budget;
}

/* Keep only the k largest logits; everything else is masked to -infinity. */
static void apply_top_k(float *work, int V, int k) {
    if (k < 1) k = 1;
    if (k >= V) return;
    float kth = -INFINITY;
    for (int i = 0; i < k; ++i) {
        int best = -1;
        float bv = -INFINITY;
        for (int j = 0; j < V; ++j) if (isfinite(work[j]) && work[j] > bv) { bv = work[j]; best = j; }
        if (best < 0) break;
        if (i == k - 1) { kth = bv; break; }
        work[best] = -INFINITY;
    }
    if (isfinite(kth)) for (int j = 0; j < V; ++j) if (work[j] < kth) work[j] = -INFINITY;
}

static int sample_next(const float *logits, int V, const GenParams *gp, uint32_t *rng) {
    float *work = (float *)aster_xmalloc((size_t)V * sizeof(float));
    for (int i = 0; i < V; ++i) work[i] = isfinite(logits[i]) ? logits[i] : -INFINITY;

    int pick;
    if (gp->temperature <= 0.0f) {
        /* Greedy: first index wins ties, so the result is deterministic. */
        pick = 0;
        for (int i = 1; i < V; ++i) if (work[i] > work[pick]) pick = i;
        free(work);
        return pick;
    }

    const float inv_t = 1.0f / gp->temperature;
    for (int i = 0; i < V; ++i) if (isfinite(work[i])) work[i] *= inv_t;
    if (gp->top_k > 0) apply_top_k(work, V, gp->top_k);

    softmax_row(work, V);
    double total = 0.0;
    for (int i = 0; i < V; ++i) total += work[i];
    if (!(total > 0.0)) { free(work); return 0; }

    double r = (double)aster_rng_uniform(rng) * total;
    double acc = 0.0;
    pick = V - 1;
    for (int i = 0; i < V; ++i) { acc += work[i]; if (r <= acc) { pick = i; break; } }
    free(work);
    return pick;
}

size_t aster_generate(AsterModel *m, const char *prompt, const GenParams *gp,
                      char *out, size_t cap, int *hit_end, int *truncated) {
    const AsterConfig *cfg = &m->cfg;
    const int C = cfg->context;

    int max_new = gp->max_new_tokens;
    if (max_new < 1) max_new = 1;
    if (max_new > C - 1) max_new = C - 1;
    const char *system = gp->system ? gp->system : "";
    int budget = aster_prompt_budget(cfg, max_new, (int)strlen(system));

    /* Encode the prompt and drop the OLDEST bytes when it is too long, so the
     * most recent part of the question survives. */
    TokenList pl;
    token_list_init(&pl);
    if (prompt && *prompt) tok_encode(prompt, strlen(prompt), &pl);
    if ((int)pl.count > budget) {
        pl.count -= (size_t)((int)pl.count - budget);
        if (truncated) *truncated = 1;
    }

    /* Calloc'd: the arena is sized for the longest possible sequence and the
     * unused tail is never read, but zero keeps the bounds check in the
     * forward pass meaningful. */
    uint16_t *seq = (uint16_t *)aster_xcalloc((size_t)C, sizeof(uint16_t));
    int n = 0;
    seq[n++] = TOK_SYSTEM;
    if (*system) {
        TokenList sl;
        token_list_init(&sl);
        tok_encode(system, strlen(system), &sl);
        for (size_t i = 0; i < sl.count && n < C; ++i) seq[n++] = sl.ids[i];
        token_list_free(&sl);
    }
    seq[n++] = TOK_USER;
    for (size_t i = 0; i < pl.count; ++i) seq[n++] = pl.ids[i];
    seq[n++] = TOK_END;
    seq[n++] = TOK_ASSISTANT;
    token_list_free(&pl);
    if (n > C) n = C;

    uint32_t rng;
    aster_rng_seed_init(&rng, gp->seed ? gp->seed : 0x5EEDu);

    AsterActs *a = aster_acts_new(1, n, cfg);
    char *raw = (char *)aster_xmalloc((size_t)C * 4 + 8);
    size_t raw_n = 0;
    if (hit_end) *hit_end = 0;

    for (int step = 0; step < max_new; ++step) {
        if (aster_forward(m, a, seq, NULL, NULL, NULL) < 0.0) break;
        /* The arena is sized for the longest possible sequence, so the last
         * row of logits is row (n - 1), not row (a->T - 1). */
        const float *lg = a->logits + (size_t)(n - 1) * (size_t)cfg->vocab;
        int next = sample_next(lg, cfg->vocab, gp, &rng);
        if (next == TOK_END || next == TOK_PAD) { if (hit_end) *hit_end = 1; break; }
        /* A structural marker means the model tried to leave the assistant
         * turn. Stop rather than loop: re-running the forward pass on an
         * unchanged sequence would just resample the same marker and burn the
         * step budget with nothing to show for it. */
        if (next > ASTER_BYTE_MAX) { if (truncated) *truncated = 1; break; }
        if (n >= C) { if (truncated) *truncated = 1; break; }
        seq[n++] = (uint16_t)next;
        raw[raw_n++] = (char)(unsigned char)next;
    }

    /* The model emits raw bytes, so validate UTF-8 before anything else
     * touches the text. */
    size_t written = utf8_sanitize(raw, raw_n, out, cap);
    free(raw);
    free(seq);
    aster_acts_free(a);
    return written;
}
