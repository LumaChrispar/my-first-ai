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
    char         hash[65]; /* SHA-256 over the sources */
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

void train_config_default(TrainConfig *c);
AsterDataset *dataset_build(const DataSpec *spec, char *err, size_t errlen);
void          dataset_free(AsterDataset *d);

/* Mean weighted cross-entropy over a dataset. Returns -1 on error. */
double dataset_eval_loss(AsterModel *m, AsterDataset *d, int batch, double *out_ppl);

/* Trains and writes a checkpoint plus a sidecar <out_path>.meta.json file.
 * Returns 0 on success, non-zero on failure with the reason in err. */
int aster_train(const TrainConfig *cfg, const AsterConfig *mcfg,
                const DataSpec *train_spec, const DataSpec *val_spec,
                const char *out_path, char *err, size_t errlen);

#endif /* ASTER_TRAIN_H */
