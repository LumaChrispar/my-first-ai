#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "model.h"
static AsterModel *M; static AsterActs *A;
static const uint16_t X[4]={1,5,9,20}, Y[4]={5,9,20,42};
static const float Wt[4]={1,1,1,1};
static double L(void){ double l=0; aster_forward(M,A,X,Y,Wt,&l); return l; }
static double D(int i,double h){ const float o=M->params[i];
    M->params[i]=(float)(o+h); double p=L();
    M->params[i]=(float)(o-h); double q=L();
    M->params[i]=o; return (p-q)/(2.0*h); }
/* R with `lv` levels of halving starting at h0 */
static double R(int i,double h0,int lv){ double a=D(i,h0);
    for(int k=1;k<lv;++k){ double b=D(i,h0*ldexp(1.0,-k)); a=(4.0*b-a)/3.0; }
    return a; }
int main(int argc,char**argv){
    int L_=(argc>1)?atoi(argv[1]):1;
    AsterConfig gc; aster_config_default(&gc);
    gc.n_layer=L_; gc.d_model=8; gc.n_head=2; gc.d_ff=16; gc.context=8; gc.vocab=261;
    M=aster_model_new(&gc,7u,1); A=aster_acts_new(1,4,&gc);
    double l0=0; aster_forward(M,A,X,Y,Wt,&l0);
    memset(M->grads,0,sizeof(float)*(size_t)M->off.total);
    aster_backward(M,A,X,Y,Wt);
    int n=M->off.total;
    double *num=malloc(sizeof(double)*(size_t)n);
    for(int i=0;i<n;++i) num[i]=D(i,1e-2);
    double mx=0; for(int i=0;i<n;++i) if(fabs(num[i])>mx) mx=fabs(num[i]);
    double cut=1e-2*mx; int ncmp=0;
    for(int i=0;i<n;++i) if(fabs(num[i])>=cut) ++ncmp;
    printf("L=%d total=%d maxnum=%.4e cut=%.4e compared=%d%c",L_,n,mx,cut,ncmp,10);
    double h0s[5]={1e-2,ldexp(1.0,-7),ldexp(1.0,-8),ldexp(1.0,-9),ldexp(1.0,-10)};
    for(int f=0;f<5;++f){
        for(int lv=2;lv<=4;++lv){
            double worst=0; int wi=-1, over=0;
            for(int i=0;i<n;++i){
                if(fabs(num[i])<cut) continue;
                double v=R(i,h0s[f],lv);
                double r=fabs((double)M->grads[i]-v)/fabs(v);
                if(r>worst){worst=r;wi=i;}
                if(r>0.02)++over;
            }
            printf("  h0=%9.3e lv=%d  worst_rel=%.4f  over_2pct=%3d  (worst param %d)%c",
                   h0s[f],lv,worst,over,wi,10);
        }
    }
    return 0;
}
