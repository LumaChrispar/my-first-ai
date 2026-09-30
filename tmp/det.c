#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "model.h"
int main(void) {
    AsterConfig gc; aster_config_default(&gc);
    gc.n_layer=1; gc.d_model=8; gc.n_head=2; gc.d_ff=16; gc.context=8; gc.vocab=261;
    AsterModel *m = aster_model_new(&gc, 7u, 1);
    AsterActs *a = aster_acts_new(1, 4, &gc);
    const uint16_t gx[4]={1,5,9,20}, gy[4]={5,9,20,42};
    const float gw[4]={1,1,1,1};
    double l; aster_forward(m,a,gx,gy,gw,&l);
    printf("forward repeated, no writes between:%c", 10);
    for (int k=0;k<5;++k){ double t; aster_forward(m,a,gx,gy,gw,&t);
        printf("  run %d: %.17g  (bitwise same as first: %s)%c", k, t,
               t==l?"yes":"NO", 10); }
    printf("%c", 10);
    memset(m->grads,0,sizeof(float)*(size_t)m->off.total);
    aster_backward(m,a,gx,gy,gw);
    double t2; aster_forward(m,a,gx,gy,gw,&t2);
    printf("after backward: %.17g  same as first: %s%c", t2, t2==l?"yes":"NO", 10);
    /* now touch an unread pos_emb row, then re-check the loss */
    float o = m->params[m->off.pos_emb + 5*8];
    m->params[m->off.pos_emb + 5*8] = o + 1.0f;
    double t3; aster_forward(m,a,gx,gy,gw,&t3);
    m->params[m->off.pos_emb + 5*8] = o;
    printf("perturbed pos_emb[5]: %.17g  same as first: %s  (%.17g)%c",
           t3, t3==l?"yes":"NO", t3-l, 10);
    return 0;
}
