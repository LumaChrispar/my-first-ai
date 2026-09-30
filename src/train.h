/* train.h - dataset construction, the training loop, and evaluation.
 *
 * DATA REPRESENTATION
 * -------------------
 * A dataset is one token stream plus a list of windows into it. Windows are
 * gathered into batches on demand, so a large corpus does not have to be
 * materialised as examples x batch x block. Each window carries a per-
 * position weight: 1.0 where the position should contribute to the loss and
 * 0.0 for prompt tokens we want the model to see but not to be trained on.
 */
#ifndef ASTER_TRAIN_H
#define ASTER_TRAIN_H

#include <stddef.h>
#include <stdint.h>
#include "model.h"

typedef enum { ASTER_DATA_TEXT = 0, ASTER_DATA_CHAT = 1 } AsterDataMode;

typedef struct {
    const char **paths;
    int          n_paths;
    AsterDataMode mode;
    int          block;        /* tokens per training window */
    int          max_examples; /* 0 means no cap */
    const AsterVocab *v;       /* tokenizer; must match the model's */
} DataSpec;

typedef struct {
    size_t start;   /* offset into the shared token stream */
    int    len;     /* tokens in this window, <= block */
    const float *w;  /* per-position loss weight, len entries */
} AsterWindow;

typedef struct {
    uint16_t    *stream;
    size_t       stream_len;
    AsterWindow *win;
    int          n_win;
    float       *ones;     /* shared all-ones weights for unmasked data */
    float       *wbuf;     /* owned per-window weights, or NULL */
    char         hash[65]; /* SHA-256 of the source bytes, hex */
    size_t       source_bytes;
    size_t       tokens;    /* tokens in the stream */
    int          docs;      /* source files or conversations */
    int          truncated; /* windows shortened to fit the block */
    int          block;
} AsterDataset;

typedef struct {
    int      batch;
    int      steps;
    int      log_every;
    int      warmup;
    float    lr;
    float    min_ratio;      /* cosine floor as a fraction of lr */
    float    weight_decay;
    float    beta1, beta2, eps;
    float    clip;           /* global-norm gradient clipping */
    uint32_t seed;
} TrainConfig;

/* Loss, reported two ways because only one of them survives a tokenizer
 * change.
 *
 * nats_per_token falls whenever tokens get easier, and a sub-word tokenizer
 * makes every token easier whether or not the model got better at anything.
 * It is comparable only against another model using the SAME vocabulary.
 *
 * bits_per_byte is the number to quote when comparing across tokenizers. It
 * weights each target by the bytes that target covers, so the two figures
 * differ by exactly the compression ratio and nothing else. */
typedef struct {
    double nats_per_token;
    double nats_per_byte;
    double bits_per_byte;    /* nats_per_byte / ln 2 -- a division, not exp() */
    double token_ppl;        /* exp(nats_per_token); label it, never compare it */
    size_t tokens, bytes, weight;
} AsterLoss;

void train_config_default(TrainConfig *c);
AsterDataset *dataset_build(const DataSpec *spec, char *err, size_t errlen);
void          dataset_free(AsterDataset *d);

/* Full loss breakdown over a dataset. Returns -1 on error. */
double dataset_eval_loss_full(AsterModel *m, AsterDataset *d, int batch, AsterLoss *out);

/* Mean weighted cross-entropy per token. Returns -1 on error. */
double dataset_eval_loss(AsterModel *m, AsterDataset *d, int batch, double *out_ppl);

/* Walks every span of text the model would be trained on, in order, calling
 * fn for each. This is the seam the vocabulary learner uses, so the merges
 * are fitted on exactly the text the model sees rather than on the raw file
 * including its JSON punctuation.
 *
 * There is deliberately no way to pass a validation file here. A merge table
 * fitted on held-out data leaks it, and the held-out loss then stops meaning
 * anything -- so the option is not offered rather than offered and warned
 * about. */
typedef void (*AsterTextFn)(void *ud, const char *s, size_t n);
int dataset_scan_text(const DataSpec *spec, AsterTextFn fn, void *ud, char *err, size_t errlen);

/* Trains and writes a checkpoint plus a sidecar <out_path>.meta.json file.
 * Returns 0 on success, non-zero on failure with the reason in err. */
int aster_train(const TrainConfig *cfg, const AsterConfig *mcfg,
                const DataSpec *train_spec, const DataSpec *val_spec,
                const char *out_path, char *err, size_t errlen);

#endif /* ASTER_TRAIN_H */
