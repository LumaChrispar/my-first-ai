/* Throwaway finite-difference gradient check (not part of the build).
 *
 * The hard part is not the finite difference, it is knowing how much of the
 * numeric answer is float32 noise. The loss is O(5) and is accumulated in
 * float32, so f(x) carries an absolute error of roughly 1e-6; divided by 2h
 * that is a noise floor of ~1e-6/(2h) in the gradient. With h=1e-3 the floor is
 * ~5e-4, which is the same size as the gradients we are trying to measure --
 * so a naive check cannot tell "wrong by 10x" from "noise".
 *
 * Two things make the check trustworthy:
 *   1. Richardson extrapolation from h and h/2 kills the O(h^2) truncation
 *      term without needing a smaller, noisier step.
 *   2. The noise floor is MEASURED, not guessed: pos_emb rows at positions
 *      >= T are never read by the forward pass, so their true gradient is
 *      exactly zero and the finite difference of them is pure noise.
 *
 * Usage: gradcheck <n_layer> <d_model> <n_head> <T>
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "model.h"

static AsterModel *M;
static AsterActs  *A;
static uint16_t    X[16], Y[16];
static float       Wt[16];

static double loss_at(const uint16_t *x) {
    double l = 0.0;
    aster_forward(M, A, x, Y, Wt, &l);
    return l;
}

/* Richardson-extrapolated central difference. */
static double fd(int i) {
    const float o = M->params[i];
    double h = 1e-2, acc = 0.0, prev = 0.0;
    for (int k = 0; k < 2; ++k) {
        M->params[i] = (float)(o + h);  double p = loss_at(X);
        M->params[i] = (float)(o - h);  double m = loss_at(X);
        double d = (p - m) / (2.0 * h);
        acc = (k == 0) ? d : (4.0 * d - prev) / 3.0;   /* (4*D(h/2) - D(h))/3 */
        prev = d;
        h *= 0.5;
    }
    M->params[i] = o;
    return acc;
}

static const char *region(int i, int L, int *which) {
    const AsterOffsets *o = &M->off;
    const AsterConfig  *c = &M->cfg;
    *which = 0;
    if (i < o->tok_emb + c->vocab * c->d_model) return "tok_emb";
    int lstart = o->tok_emb + c->vocab * c->d_model + c->context * c->d_model;
    if (i < lstart) return "pos_emb";
    int l = (i - lstart) / o->per_layer;
    if (l >= L) l = L - 1;
    int base = lstart + l * o->per_layer, k = i - base;
    if (k < o->lw_wq) return (k < o->lw_ln1b) ? "ln1_w" : "ln1_b";
    if (k < o->lw_wv) return (k < o->lw_wk) ? "Wq" : "Wk";
    if (k < o->lw_wo) return "Wv";
    if (k < o->lw_ln2w) return "Wo";
    if (k < o->lw_ln2b) return "ln2_w";
    if (k < o->lw_ff1) return "ln2_b";
    if (k < o->lw_ff2) return "ff1+bias";
    return "ff2";
}

int main(int argc, char **argv) {
    int L  = (argc > 1) ? atoi(argv[1]) : 1;
    int C  = (argc > 2) ? atoi(argv[2]) : 8;
    int H  = (argc > 3) ? atoi(argv[3]) : 1;
    int T  = (argc > 4) ? atoi(argv[4]) : 6;
    /* Subsystem ablation: zero whole blocks so the remaining path is one whose
     * correct gradient can be reasoned about by hand. Bisecting beats staring. */
    int MODE = (argc > 5) ? atoi(argv[5]) : 0;
    if (T < 1) T = 1;
    if (T > 8) T = 8;

    AsterConfig cfg;
    aster_config_default(&cfg);
    cfg.n_layer = L; cfg.n_head = H; cfg.d_model = C; cfg.d_ff = 2 * C;
    cfg.context = 8; cfg.vocab = 261;
    char err[192];
    if (aster_config_validate(&cfg, err, sizeof err) != 0) { printf("bad cfg: %s\n", err); return 1; }

    M = aster_model_new(&cfg, 7u, 1);
    A = aster_acts_new(1, T, &cfg);

    for (int l = 0; l < L; ++l) {
        int bse = M->off.layer0 + l * M->off.per_layer;
        int n4 = 4 * cfg.d_model * cfg.d_model;
        switch (MODE) {
        case 1:   /* no attention at all */
            for (int k = 0; k < n4; ++k) M->params[bse + M->off.lw_wq + k] = 0.0f;
            break;
        case 2:   /* no feed-forward */
            for (int k = 0; k < cfg.d_ff * cfg.d_model; ++k)
                M->params[bse + M->off.lw_ff1 + k] = 0.0f;
            for (int k = 0; k < cfg.d_model * cfg.d_ff; ++k)
                M->params[bse + M->off.lw_ff2 + k] = 0.0f;
            break;
        case 3:   /* attention with uniform scores: Wo and Wv only */
            for (int k = 0; k < 2 * cfg.d_model * cfg.d_model; ++k)
                M->params[bse + M->off.lw_wq + k] = 0.0f;
            break;
        case 4:   /* feed-forward only: attention output projection is identity-ish */
            for (int k = 0; k < cfg.d_model * cfg.d_model; ++k)
                M->params[bse + M->off.lw_wo + k] = 0.0f;
            break;
        }
    }

    const uint16_t seq[8] = {1, 5, 9, 20, 100, 7, 3, 77};
    for (int i = 0; i < T; ++i) X[i] = seq[i];
    for (int i = 0; i + 1 < T; ++i) { Y[i] = X[i + 1]; Wt[i] = 1.0f; }
    Y[T - 1] = 42; Wt[T - 1] = 1.0f;

    double loss = loss_at(X);
    memset(M->grads, 0, sizeof(float) * (size_t)M->off.total);
    aster_backward(M, A, X, Y, Wt);
    printf("L=%d C=%d H=%d T=%d hd=%d loss=%.6f params=%d\n",
           L, C, H, T, M->head_dim, loss, M->off.total);

    /* --- measured noise floor from params the forward never reads --------- */
    double nsum = 0.0; int nn = 0;
    for (int t = T; t < cfg.context; ++t)
        for (int c = 0; c < C; ++c) {
            double v = fd(M->off.pos_emb + t * C + c);
            nsum += v * v; ++nn;
        }
    double noise = (nn > 0) ? sqrt(nsum / nn) : 0.0;
    double tol_sig = (noise * 4.0 > 1e-4) ? noise * 4.0 : 1e-4;
    printf("FD noise floor (mean |d| on unread pos_emb) = %.3e  -> sig threshold %.3e\n",
           noise, tol_sig);

    struct { const char *name; int bad, sig, silent; double worst; } reg[64];
    memset(reg, 0, sizeof reg);
    for (int i = 0; i < M->off.total; ++i) {
        double num = fd(i), ana = M->grads[i];
        if (fabs(num) <= tol_sig && fabs(ana) <= tol_sig) continue;
        int which; const char *nm = region(i, L, &which);
        int k = 0;
        while (k < 63 && reg[k].name && strcmp(reg[k].name, nm)) ++k;
        if (k == 63) continue;
        if (!reg[k].name) {
            char nb[48];
            snprintf(nb, sizeof nb, which > 0 ? "%s/L%d" : "%s", nm, which);
            reg[k].name = strdup(nb);
        }
        reg[k].sig++;
        double rel = fabs(ana - num) / (fabs(num) + 1e-30);
        if (rel > reg[k].worst) reg[k].worst = rel;
        if (fabs(ana - num) > 0.05 * fabs(num) + tol_sig) {
            reg[k].bad++;
            if (fabs(ana) <= tol_sig) reg[k].silent++;
        }
    }
    /* --- spot checks, so a failure is a number and not just a count ----- */
    {
        struct { const char *nm; int base; } sp[10];
        int ns = 0;
        sp[ns].nm = "pos_emb";  sp[ns++].base = M->off.pos_emb;
        sp[ns].nm = "ln1_w";    sp[ns++].base = M->off.layer0 + M->off.lw_ln1w;
        sp[ns].nm = "ln1_b";    sp[ns++].base = M->off.layer0 + M->off.lw_ln1b;
        sp[ns].nm = "Wq";       sp[ns++].base = M->off.layer0 + M->off.lw_wq;
        sp[ns].nm = "Wo";       sp[ns++].base = M->off.layer0 + M->off.lw_wo;
        sp[ns].nm = "ln2_w";    sp[ns++].base = M->off.layer0 + M->off.lw_ln2w;
        sp[ns].nm = "ff1+b";    sp[ns++].base = M->off.layer0 + M->off.lw_ff1;
        sp[ns].nm = "ff2";      sp[ns++].base = M->off.layer0 + M->off.lw_ff2;
        sp[ns].nm = "lnf_w";    sp[ns++].base = M->off.lnf_w;
        sp[ns].nm = "lnf_b";    sp[ns++].base = M->off.lnf_b;
        for (int r = 0; r < ns; ++r) {
            printf("\n%s (base %d):\n", sp[r].nm, sp[r].base);
            for (int k = 0; k < 4; ++k) {
                int i = sp[r].base + k;
                double n = fd(i);
                printf("  [%d] %d  ana %+.6e  num %+.6e  ratio %8.4f\n",
                       k, i, M->grads[i], n,
                       (fabs(M->grads[i]) > 1e-12 && fabs(n) > 1e-12)
                           ? (double)M->grads[i] / n : 0.0);
            }
        }
    }

    printf("\n%-16s %6s %6s %8s %10s\n", "region", "bad", "sig", "silent", "worst_rel");
    int badtot = 0, sigtot = 0;
    for (int k = 0; k < 63 && reg[k].name; ++k) {
        printf("%-16s %6d %6d %8d %10.4f\n",
               reg[k].name, reg[k].bad, reg[k].sig, reg[k].silent, reg[k].worst);
        badtot += reg[k].bad; sigtot += reg[k].sig;
    }
    printf("TOTAL bad=%d of %d significant\n", badtot, sigtot);

    aster_acts_free(A);
    aster_model_free(M);
    return badtot ? 1 : 0;
}
