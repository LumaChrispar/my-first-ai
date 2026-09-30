#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "model.h"
static AsterModel *M; static AsterActs *A;
static const uint16_t X[4]={1,5,9,20}, Y[4]={5,9,20,42};
static const float Wt[4]={1,1,1,1};
static double L(void){ double l=0; aster_forward(M,A,X,Y,Wt,&l); return l; }
int main(int argc,char**argv){
    int P = (argc>1)?atoi(argv[1]):167;
    AsterConfig gc; aster_config_default(&gc);
    gc.n_layer=1; gc.d_model=8; gc.n_head=2; gc.d_ff=16; gc.context=8; gc.vocab=261;
    M = aster_model_new(&gc,7u,1); A = aster_acts_new(1,4,&gc);
    double l0=0; aster_forward(M,A,X,Y,Wt,&l0);
    memset(M->grads,0,sizeof(float)*(size_t)M->off.total);
    aster_backward(M,A,X,Y,Wt);
    const float o = M->params[P];
    printf("param %d = %.9g   analytic grad = %+.8e   loss0 = %.17g%c", P, o, M->grads[P], l0, 10);
    double d[24]; double h[24]; int nh=0;
    for (int k=0;k<12;++k){
        h[nh] = ldexp(1.0, -k-2);                 /* 1/4,1/8,...,1/16384 */
        M->params[P]=(float)(o+h[nh]); double p=L();
        M->params[P]=(float)(o-h[nh]); double q=L();
        d[nh]=(p-q)/(2.0*h[nh]);
        printf("  h=%9.2e  p=%.17g  q=%.17g  p-q=%+.6e  D=%+.8e%c", h[nh], p, q, p-q, d[nh], 10);
        ++nh;
    }
    M->params[P]=o;
    /* Richardson from consecutive pairs, several levels */
    printf("%c",10);
    for (int k=0;k+1<nh;++k){
        double r=(4.0*d[k+1]-d[k])/3.0;
        printf("  R(%9.2e,%9.2e) = %+.8e   rel err vs analytic %.4f%c",
               h[k],h[k+1],r,fabs(r-M->grads[P])/fabs(M->grads[P]),10);
    }
    return 0;
}
