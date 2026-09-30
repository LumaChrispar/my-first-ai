#include <stdio.h>
#include "model.h"
int main(void) {
    AsterConfig cfg; aster_config_default(&cfg);
    cfg.n_layer=1; cfg.n_head=1; cfg.d_model=8; cfg.d_ff=16; cfg.context=8; cfg.vocab=261;
    AsterModel *M = aster_model_new(&cfg, 7u, 1);
    AsterActs *A = aster_acts_new(1, 6, &cfg);
    printf("off.total=%d params=%p grads=%p\n", M->off.total, (void*)M->params, (void*)M->grads);
    printf("xin=%p\n", (void*)A->xin);
    #define OFF(f) printf("  %-8s +%8ld\n", #f, (long)((char*)A->f - (char*)A->xin))
    OFF(ln1_out); OFF(qkv); OFF(att); OFF(attp); OFF(proj); OFF(res1); OFF(ln2_out);
    OFF(ffz); OFF(ffh); OFF(ffo); OFF(lnf_out); OFF(logits); OFF(dxin); OFF(dres);
    OFF(dln1); OFF(datt); OFF(dproj); OFF(dln2); OFF(dffh); OFF(dqkv); OFF(dlnf); OFF(dlogits);
    int bse=M->off.layer0;
    printf("zeroing params[%d..%d] = Wq,Wk,Wv,Wo of layer 0\n",
           bse+M->off.lw_wq, bse+M->off.lw_wq+255);
    for (int k=0;k<256;++k) M->params[bse+M->off.lw_wq+k]=0.0f;
    printf("P[2168..2175] ="); for (int k=0;k<8;++k) printf(" %g", M->params[2168+k]); printf("\n");
    uint16_t X[8]={1,5,9,20,100,7}, Y[8]; float Wt[8];
    for (int i=0;i+1<6;++i){Y[i]=X[i+1];Wt[i]=1.0f;}
    Y[5]=42; Wt[5]=1.0f;
    double l=0; aster_forward(M,A,X,Y,Wt,&l);
    printf("qkv[0..7]   ="); for (int k=0;k<8;++k) printf(" %g", A->qkv[k]); printf("\n");
    printf("att[0..3]   ="); for (int k=0;k<4;++k) printf(" %g", A->att[k]); printf("\n");
    printf("proj[0..3]  ="); for (int k=0;k<4;++k) printf(" %g", A->proj[k]); printf("\n");
    printf("res1[0..3]  ="); for (int k=0;k<4;++k) printf(" %g", A->res1[k]); printf("\n");
    printf("ln1_out[0..3]="); for (int k=0;k<4;++k) printf(" %g", A->ln1_out[k]); printf("\n");
    printf("ln2_out[0..3]="); for (int k=0;k<4;++k) printf(" %g", A->ln2_out[k]); printf("\n");
    printf("ffo[0..3]   ="); for (int k=0;k<4;++k) printf(" %g", A->ffo[k]); printf("\n");
    return 0;
}
