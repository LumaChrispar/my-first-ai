#include <stdio.h>
#include "model.h"
int main(void) {
    AsterConfig cfg; aster_config_default(&cfg);
    cfg.n_layer=1; cfg.n_head=1; cfg.d_model=8; cfg.d_ff=16; cfg.context=8; cfg.vocab=261;
    AsterModel *M = aster_model_new(&cfg, 7u, 1);
    AsterActs *A = aster_acts_new(1, 3, &cfg);
    uint16_t X[8]={1,5,9}, Y[8]; float Wt[8];
    for (int i=0;i+1<3;++i){Y[i]=X[i+1];Wt[i]=1.0f;}
    Y[2]=42; Wt[2]=1.0f;
    double l=0; aster_forward(M,A,X,Y,Wt,&l);
    #define ROW(nm,buf,Cn) do{ printf("%-8s", nm); \
        for (int c=0;c<Cn;++c) printf(" %12.6g", (buf)[c]); printf("\n"); }while(0)
    printf("d_model=%d\n", cfg.d_model);
    ROW("xin[0]",   A->xin, 8);
    ROW("ln1_out",  A->ln1_out, 8);
    ROW("qkv(q)",   A->qkv, 8);
    ROW("att",      A->att, 8);
    ROW("proj",     A->proj, 8);
    ROW("res1",     A->res1, 8);
    ROW("ln2_out",  A->ln2_out, 8);
    printf("ln1w="); for (int c=0;c<8;++c) printf(" %g", M->params[M->off.layer0+M->off.lw_ln1w+c]); printf("\n");
    printf("ln1b="); for (int c=0;c<8;++c) printf(" %g", M->params[M->off.layer0+M->off.lw_ln1b+c]); printf("\n");
    return 0;
}
