#include <stdio.h>
#include <stdlib.h>
#include "model.h"
int main(void) {
    AsterConfig cfg; aster_config_default(&cfg);
    cfg.n_layer=1; cfg.n_head=1; cfg.d_model=8; cfg.d_ff=16; cfg.context=8; cfg.vocab=261;
    AsterModel *M = aster_model_new(&cfg, 7u, 1);
    AsterActs *A = aster_acts_new(1, 6, &cfg);
    int bse = M->off.layer0, n4 = 4*64;
    printf("layer0=%d lw_ln1w=%d lw_ln1b=%d lw_wq=%d lw_wk=%d lw_wv=%d lw_wo=%d per_layer=%d\n",
           bse, M->off.lw_ln1w, M->off.lw_ln1b, M->off.lw_wq, M->off.lw_wk, M->off.lw_wv, M->off.lw_wo, M->off.per_layer);
    for (int k=0;k<n4;++k) M->params[bse+M->off.lw_wq+k] = 0.0f;
    printf("after zero: Wq[0]=%g Wq[63]=%g  ln1w[0]=%g\n",
           M->params[bse+M->off.lw_wq], M->params[bse+M->off.lw_wq+63], M->params[bse+M->off.lw_ln1w]);
    uint16_t X[8]={1,5,9,20,100,7}, Y[8]; float Wt[8];
    for (int i=0;i+1<6;++i){Y[i]=X[i+1];Wt[i]=1.0f;}
    Y[5]=42; Wt[5]=1.0f;
    double l0=0.0; aster_forward(M,A,X,Y,Wt,&l0);
    float o = M->params[bse+M->off.lw_ln1w];
    M->params[bse+M->off.lw_ln1w] = o + 1.0f;
    double l1=0.0; aster_forward(M,A,X,Y,Wt,&l1);
    M->params[bse+M->off.lw_ln1w] = o;
    double l2=0.0; aster_forward(M,A,X,Y,Wt,&l2);
    printf("loss base=%.10f  ln1w[0]+1 -> %.10f  restored=%.10f   delta=%.3e\n", l0, l1, l2, l1-l0);
    printf("qkv after fwd: q0=%g q1=%g  att0=%g  proj0=%g  res10=%g\n",
           A->qkv[0], A->qkv[1], A->att[0], A->proj[0], A->res1[0]);
    return 0;
}
