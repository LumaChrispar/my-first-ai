/* Does the numeric gradient converge as h -> 0, or is it dominated by
 * float32 rounding in the loss? */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "model.h"
int main(int argc, char **argv) {
    int L=1,C=8,H=1,T=6;
    AsterConfig cfg; aster_config_default(&cfg);
    cfg.n_layer=L; cfg.n_head=H; cfg.d_model=C; cfg.d_ff=2*C; cfg.context=8; cfg.vocab=261;
    AsterModel *M = aster_model_new(&cfg, 7u, 1);
    AsterActs *A = aster_acts_new(1, T, &cfg);
    uint16_t X[8]={1,5,9,20,100,7,3,77}, Y[8]; float Wt[8];
    for (int i=0;i<T;++i) X[i]=X[i];
    for (int i=0;i+1<T;++i){Y[i]=X[i+1];Wt[i]=1.0f;}
    Y[T-1]=42; Wt[T-1]=1.0f;
    double lo=0.0; aster_forward(M,A,X,Y,Wt,&lo);
    printf("loss=%.8f\n", lo);
    struct { const char*nm; int i; } t[5];
    t[0].nm="ln1w[0]"; t[0].i=M->off.layer0+M->off.lw_ln1w;
    t[1].nm="ln1b[0]"; t[1].i=M->off.layer0+M->off.lw_ln1b;
    t[2].nm="pos_emb[0]"; t[2].i=M->off.pos_emb;
    t[3].nm="lnf_w[0]"; t[3].i=M->off.lnf_w;
    t[4].nm="Wq[0]"; t[4].i=M->off.layer0+M->off.lw_wq;
    double hs[7]={1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4};
    for (int k=0;k<5;++k){
        printf("%-12s ana %+.4e  ", t[k].nm, M->grads[t[k].i]);
        for (int j=0;j<7;++j){
            float o=M->params[t[k].i];
            M->params[t[k].i]=(float)(o+hs[j]); double p=0.0; aster_forward(M,A,X,Y,Wt,&p);
            M->params[t[k].i]=(float)(o-hs[j]); double m=0.0; aster_forward(M,A,X,Y,Wt,&m);
            M->params[t[k].i]=o;
            printf("%+.3e ", (p-m)/(2.0*hs[j]));
        }
        printf("\n");
    }
    printf("%-12s          ", "h=");
    for (int j=0;j<7;++j) printf("%-8.0e ", hs[j]);
    printf("\n");
    return 0;
}
