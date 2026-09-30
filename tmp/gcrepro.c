#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "model.h"
static double gc_loss(AsterModel *m, AsterActs *a, const uint16_t *x,
                      const uint16_t *y, const float *w) {
    double l = 0.0; aster_forward(m, a, x, y, w, &l); return l;
}
static double gc_numeric(AsterModel *m, AsterActs *a, const uint16_t *x,
                         const uint16_t *y, const float *w, int i) {
    const float o = m->params[i];
    double h = 1e-2, d = 0.0, prev = 0.0;
    for (int k = 0; k < 2; ++k) {
        m->params[i] = (float)(o + h); double p = gc_loss(m,a,x,y,w);
        m->params[i] = (float)(o - h); double q = gc_loss(m,a,x,y,w);
        d = (k == 0) ? (p - q)/(2.0*h) : (4.0*d - prev)/3.0;
        prev = (p - q)/(2.0*h);
        h *= 0.5;
    }
    m->params[i] = o;
    return d;
}
int main(void) {
    AsterConfig gc; aster_config_default(&gc);
    gc.n_layer=1; gc.d_model=8; gc.n_head=2; gc.d_ff=16; gc.context=8; gc.vocab=261;
    AsterModel *m = aster_model_new(&gc, 7u, 1);
    AsterActs *a = aster_acts_new(1, 4, &gc);
    const uint16_t gx[4]={1,5,9,20}, gy[4]={5,9,20,42};
    const float gw[4]={1,1,1,1};
    double l0=0; aster_forward(m,a,gx,gy,gw,&l0);
    memset(m->grads,0,sizeof(float)*(size_t)m->off.total);
    aster_backward(m,a,gx,gy,gw);
    int n = m->off.total;
    double maxnum=0;
    double *num = malloc(sizeof(double)*(size_t)n);
    for (int i=0;i<n;++i){ num[i]=gc_numeric(m,a,gx,gy,gw,i); if(fabs(num[i])>maxnum) maxnum=fabs(num[i]); }
    printf("total=%d pos_emb=%d loss=%.6f maxnum=%.4e%c", n, m->off.pos_emb, l0, maxnum, 10);
    for (int i=0;i<n;++i) if (fabs(num[i]) >= 1e-2*maxnum) {
        double r = fabs((double)m->grads[i]-num[i])/fabs(num[i]);
        if (r > 0.02) printf("  BAD i=%d ana %+.6e num %+.6e rel=%.4f%c", i, m->grads[i], num[i], r, 10);
    }
    printf("param 2113: ana %+.6e num %+.6e  pos_emb[3][1]%c", m->grads[2113], num[2113], 10);
    return 0;
}
