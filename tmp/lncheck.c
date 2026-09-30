/* Isolate layer_norm_bwd and check it against a central difference of
 * layer_norm_fwd. No model, no residual stream, no attention. */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define ASTER_LN_EPS 1e-5f
#define N 8
static const float h = 1e-3f;

static void lnf(const float *x, const float *w, const float *b, float *y, int n) {
    double mu = 0, var = 0;
    for (int i = 0; i < n; ++i) mu += x[i];
    mu /= n;
    for (int i = 0; i < n; ++i) { double d = x[i] - mu; var += d * d; }
    var /= n;
    double inv = 1.0 / sqrt(var + (double)ASTER_LN_EPS);
    for (int i = 0; i < n; ++i) y[i] = (float)(((x[i] - mu) * inv) * w[i] + b[i]);
}

static void lnb(const float *x, const float *dy, const float *w,
                float *dx, int n, float *dw, float *db) {
    double mu = 0, var = 0, m1 = 0, m2 = 0;
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

static double L(const float *x, const float *w, const float *b,
                const float *dy) {
    float y[N];
    lnf(x, w, b, y, N);
    double s = 0;
    for (int k = 0; k < N; ++k) s += dy[k] * y[k];
    return s;
}

static double fd(float *arr, int idx, const float *x, float *w, float *b,
                 const float *dy) {
    float *tgt = (arr == x) ? x : (arr == w) ? w : b;
    float o = tgt[idx];
    tgt[idx] = (float)(o + h); double p = L(x, w, b, dy);
    tgt[idx] = (float)(o - h); double m = L(x, w, b, dy);
    tgt[idx] = o;
    return (p - m) / (2.0 * h);
}

int main(void) {
    float x[N], w[N], b[N], dy[N], dx[N];
    float dw[N] = {0}, db[N] = {0};
    srand(12345);
    for (int i = 0; i < N; ++i) { x[i] = (float)rand() / RAND_MAX; w[i] = 1.0f; b[i] = 0.0f; }
    for (int i = 0; i < N; ++i) dy[i] = (float)rand() / RAND_MAX - 0.5f;

    lnb(x, dy, w, dx, N, dw, db);

    double wdx = 0, wdw = 0, wdb = 0;
    for (int i = 0; i < N; ++i) {
        double ndx = fd(x, i, x, w, b, dy);
        double ndw = fd(w, i, x, w, b, dy);
        double ndb = fd(b, i, x, w, b, dy);
        double ed = fabs(dx[i] - ndx) / (fabs(ndx) + 1e-12);
        double ew = fabs(dw[i] - ndw) / (fabs(ndw) + 1e-12);
        double eb = fabs(db[i] - ndb) / (fabs(ndb) + 1e-12);
        if (ed > wdx) wdx = ed;
        if (ew > wdw) wdw = ew;
        if (eb > wdb) wdb = eb;
        printf("i=%d  dx %+.6e/%+.6e  dw %+.6e/%+.6e  db %+.6e/%+.6e%c",
               i, dx[i], ndx, dw[i], ndw, db[i], ndb, 10);
    }
    printf("worst rel: dx=%.4f dw=%.4f db=%.4f%c", wdx, wdw, wdb, 10);
    return 0;
}
