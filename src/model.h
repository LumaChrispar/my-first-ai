/* model.h - decoder-only causal transformer, checkpoint IO, and generation.
 *
 * TENSOR LAYOUT (all row-major float32)
 * --------------------------------------
 * Activations are [batch][layer][time][channel] and "time" is a token
 * position. Weight matrices are [out][in], so row `o` of y = x @ W^T + b.
 * Attention is computed per [batch][layer][head] over a [time][head_dim] view.
 *
 * PARAMETER BLOCK
 * ---------------
 * All parameters live in one contiguous float array so serialization and the
 * AdamW state are trivial and there is no pointer bookkeeping to get wrong.
 * AsterOffsets is the single source of truth for the order:
 *
 *   tok_emb[vocab][d_model]          token embedding
 *   pos_emb[context][d_model]        learned position embedding
 *   per layer l (at layer0 + l*per_layer, using the lw_* offsets):
 *     ln1_w, ln1_b            [d_model]
 *     wq, wk, wv, wo          [d_model][d_model]
 *     ln2_w, ln2_b            [d_model]
 *     ff1                      [d_ff][d_model]
 *     ff2                      [d_model][d_ff]
 *   lnf_w, lnf_b              [d_model]   final layer norm
 *
 * Weight tying: the output head IS the token embedding table. The same
 * tok_emb values are read in the forward pass and receive gradient in the
 * backward pass, so the two halves can never drift apart.
 */
#ifndef ASTER_MODEL_H
#define ASTER_MODEL_H

#include <stddef.h>
#include <stdint.h>
#include "tokenizer.h"

#define ASTER_ARCH_VERSION 1
#define ASTER_CHECKPOINT_MAGIC "ASTERMD1"
#define ASTER_CHECKPOINT_FORMAT 1
#define ASTER_LN_EPS 1e-5f

/* Named defaults from the build spec. A checkpoint stores and validates its
 * own values, so changing these never corrupts an existing model file. */
#define ASTER_DEFAULT_LAYERS     2
#define ASTER_DEFAULT_DMODEL     64
#define ASTER_DEFAULT_HEADS      4
#define ASTER_DEFAULT_DFF        256
#define ASTER_DEFAULT_CTX        128
#define ASTER_DEFAULT_MAX_NEW    80

typedef struct {
    int n_layer, n_head, d_model, d_ff, context, vocab;
} AsterConfig;

typedef struct {
    int tok_emb, pos_emb, layer0, lnf_w, lnf_b;
    int lw_ln1w, lw_ln1b, lw_wq, lw_wk, lw_wv, lw_wo,
        lw_ln2w, lw_ln2b, lw_ff1, lw_ff2;
    int per_layer;   /* elements in one transformer block */
    int total;       /* total parameter count */
} AsterOffsets;

typedef struct {
    AsterConfig  cfg;
    AsterOffsets off;
    float *params;   /* [total] */
    float *grads;    /* [total] or NULL */
    int    head_dim;
} AsterModel;

/* Activations kept for the backward pass, allocated once and reused. */
typedef struct {
    int B, T, H, C, F, V, L;
    float *xin;      /* [B][L+1][T][C] block inputs; l=0 is token+pos, l=L is the final stream */
    float *ln1_out;  /* [B][L][T][C] */
    float *qkv;      /* [B][L][T][3C] */
    float *att;      /* [B][L][T][C]  concatenated head outputs */
    float *attp;     /* [B][L][H][T][T]  post-softmax attention weights */
    float *proj;     /* [B][L][T][C]  after the output projection wo */
    float *res1;     /* [B][L][T][C]  xin + proj  (the ln2 input) */
    float *ln2_out;  /* [B][L][T][C] */
    float *ffz;      /* [B][L][T][F]  pre-activation, i.e. the GELU *input* */
    float *ffh;      /* [B][L][T][F]  after GELU */
    float *ffo;      /* [B][L][T][C]  ff2 output, pre residual */
    float *lnf_out;  /* [B][T][C]  final layer norm output */
    float *logits;   /* [B][T][V] */
    /* backward scratch, one slot per (b, l, t) so nothing aliases */
    float *dxin;     /* [B][L+1][T][C] gradient on the residual stream at each stage */
    float *dres;     /* [B][L][T][C] per-layer working copy: layer l reads
                      * dxin[l+1] into here, then stores d(xin[l]) into dxin[l].
                      * Accumulating in place on dxin would clobber the value
                      * the next layer down still needs to read. */
    float *dln1;     /* [B][L][T][C] */
    float *datt;     /* [B][L][T][C] */
    float *dproj;    /* [B][L][T][C] */
    float *dln2;     /* [B][L][T][C] */
    float *dffh;     /* [B][L][T][F] */
    float *dqkv;     /* [B][L][T][3C] */
    float *dlnf;     /* [B][T][C] */
    float *dlogits;  /* [B][T][V] */
} AsterActs;

void aster_config_default(AsterConfig *cfg);
/* 0 if usable, otherwise writes a human-readable reason into err. */
int  aster_config_validate(const AsterConfig *cfg, char *err, size_t errlen);
int  aster_param_count(const AsterConfig *cfg);
void aster_offsets_init(AsterOffsets *off, const AsterConfig *cfg);

AsterModel *aster_model_new(const AsterConfig *cfg, uint32_t seed, int want_grads);
void         aster_model_free(AsterModel *m);
AsterActs  *aster_acts_new(int B, int T, const AsterConfig *cfg);
AsterActs  *aster_acts_resize(AsterActs *a, int B, int T, const AsterConfig *cfg);
void         aster_acts_free(AsterActs *a);

/* Forward pass over B sequences of T positions.
 *   x        [B][T] input token ids, each < cfg.vocab
 *   targets  [B][T] the token to predict at each position (may be NULL)
 *   weights  [B][T] 1.0 where a position contributes to the loss, 0.0 for
 *            masked prompt tokens (may be NULL -> no loss computed)
 * Returns the mean weighted cross-entropy, or -1.0 on a bounds error.
 * `loss` may be NULL when only the forward pass is needed. */
double aster_forward(AsterModel *m, AsterActs *a, const uint16_t *x,
                     const uint16_t *targets, const float *weights, double *loss);

/* Backward pass for the most recent matching aster_forward() call.
 * Accumulates into m->grads (which must be non-NULL). */
double aster_backward(AsterModel *m, AsterActs *a, const uint16_t *x,
                      const uint16_t *targets, const float *weights);

/* ---- checkpoint ---------------------------------------------------------
 * Layout (scalars written explicitly little-endian, floats as raw IEEE-754):
 *   char[8] magic "ASTERMD1"
 *   u32 format_version, arch_version, tokenizer_version
 *   u32 n_layer, n_head, d_model, d_ff, context, vocab
 *   u32 param_count
 *   u32 train_step, seed
 *   u32 has_optimizer
 *   u32 max_new_tokens_default
 *   f32 temperature_default
 *   f32 params[param_count]
 *   f32 adam_m[...], adam_v[...]   (only when has_optimizer)
 * No chat logs, prompts, or dataset contents are ever stored here. */
/* Loads a checkpoint. On success the AdamW state is returned through out_m
 * and out_v when the file carries it and the caller asked for it; the caller
 * owns and frees them. Any failure returns NULL with a reason in err and
 * leaves the model untouched. */
AsterModel *aster_checkpoint_load(const char *path,
                                  float **out_m, float **out_v,
                                  uint32_t *out_step, uint32_t *out_seed,
                                  char *err, size_t errlen);
void aster_checkpoint_save(const AsterModel *m, const char *path,
                           const float *adam_m, const float *adam_v,
                           uint32_t step, uint32_t seed,
                           uint32_t max_new_tokens, float temperature);

/* ---- generation --------------------------------------------------------- */
typedef struct {
    int      max_new_tokens;   /* hard cap on tokens produced */
    float    temperature;      /* <= 0 means greedy and fully deterministic */
    int      top_k;            /* 0 means no top-k filter; else clamped to vocab */
    uint32_t seed;
    /* Text placed between the SYSTEM and USER markers. This MUST match the
     * system text used at training time (or be empty if the corpus had no
     * system turn). A model of this size has no way to infer the system text,
     * so a mismatch shows up as empty or nonsensical replies. NULL == empty. */
    const char *system;
} GenParams;

void gen_params_default(GenParams *g);

/* Bytes of user prompt that fit once the markers, the system text, and room
 * to answer are reserved. Pass system_bytes == 0 for no system text. */
int  aster_prompt_budget(const AsterConfig *cfg, int max_new_tokens, int system_bytes);

/* Builds [SYSTEM] <system> [USER] <prompt> [END] [ASSISTANT] and samples.
 * When the prompt does not fit, the OLDEST prompt bytes are dropped and
 * *truncated is set so the caller can say so out loud. *hit_end reports that
 * generation stopped on the END marker. Returns the bytes written to `out`,
 * which is always NUL-terminated and valid UTF-8. */
size_t aster_generate(AsterModel *m, const char *prompt, const GenParams *gp,
                      char *out, size_t cap, int *hit_end, int *truncated);

#endif /* ASTER_MODEL_H */
