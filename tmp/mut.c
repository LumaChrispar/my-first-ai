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
static double R(int i){ double a=D(i,1e-2), b=D(i,5e-3); return (4.0*b-a)/3.0; }
static void snap(float *out){ memcpy(out,M->params,sizeof(float)*(size_t)M->off.total); }
static int diff(const float *a,const float *b,int n){ int c=0; double mx=0;
    for(int i=0;i<n;++i) if(a[i]!=b[i]){ ++c; double d=fabs(a[i]-b[i]); if(d>mx)mx=d; }
    printf("   changed %d/%d params, max |delta| = %.3e%c",c,n,mx,10); return c; }
int main(void){
    AsterConfig gc; aster_config_default(&gc);
    gc.n_layer=1; gc.d_model=8; gc.n_head=2; gc.d_ff=16; gc.context=8; gc.vocab=261;
    M=aster_model_new(&gc,7u,1); A=aster_acts_new(1,4,&gc);
    int n=M->off.total;
    float *s0=malloc(sizeof(float)*(size_t)n), *s1=malloc(sizeof(float)*(size_t)n);
    double l0=0; snap(s0);
    printf("after aster_forward:"); diff(s0,M->params,n); snap(s1);
    memset(M->grads,0,sizeof(float)*(size_t)n);
    printf("after memset grads:"); diff(s1,M->params,n); snap(s1);
    aster_backward(M,A,X,Y,Wt);
    printf("after aster_backward:"); diff(s1,M->params,n); snap(s1);
    double r_first = R(2113);
    printf("R(2113) immediately      = %+.8e  (ana %+.8e)%c", r_first, M->grads[2113],10);
    for(int i=0;i<n;++i) if(i!=2113) D(i,1e-2);
    printf("   after D() on every other param:"); diff(s1,M->params,n);
    double r_after = R(2113);
    printf("R(2113) after full sweep = %+.8e  (ana %+.8e)%c", r_after, M->grads[2113],10);
    return 0;
}
