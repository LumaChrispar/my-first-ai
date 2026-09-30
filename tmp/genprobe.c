#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
static float lg_at(const float *lg, int v){ return (v < 0) ? -3.0e38f : lg[v]; }
static const char *tokname(int v){
    if (v == TOK_END) return "END"; if (v == TOK_PAD) return "PAD";
    if (v == TOK_USER) return "USER"; if (v == TOK_ASSISTANT) return "ASSISTANT";
    if (v == TOK_SYSTEM) return "SYSTEM";
    if (v >= 32 && v < 127) { static char b[8]; snprintf(b,sizeof b,"'%c'",(char)v); return b; }
    return "ctrl";
}
int main(int argc,char**argv){
    float *am=NULL,*av=NULL; uint32_t st=0,sd=0; char err[160];
    AsterModel *m=aster_checkpoint_load(argv[1],&am,&av,&st,&sd,err,sizeof err);
    if(!m){ printf("load: %s\n",err); return 1; }
    const char *prompt = argc>2?argv[2]:"Who are you?";
    const AsterConfig *cfg=&m->cfg; int C=cfg->context;
    uint16_t seq[256]; int n=0;
    seq[n++]=TOK_SYSTEM; seq[n++]=TOK_USER;
    for (const char *p=prompt; *p; ++p) seq[n++]=(uint16_t)(unsigned char)*p;
    seq[n++]=TOK_END; seq[n++]=TOK_ASSISTANT;
    AsterActs *a=aster_acts_new(1,n,cfg);
    printf("step %d  train_step=%u  prompt=%s%c",0,st,prompt,10);
    for(int step=0; step<20 && n<C; ++step){
        aster_forward(m,a,seq,NULL,NULL,NULL);
        const float *lg=a->logits+(size_t)(n-1)*cfg->vocab;
        int top[5];
        for(int k=0;k<5;++k) top[k]=-1;
        for(int v=0;v<cfg->vocab;++v){
            int p=4;
            while(p>0 && lg_at(lg,v) > lg_at(lg,top[p-1])){ top[p]=top[p-1]; --p; }
            if (lg_at(lg,v) > lg_at(lg,top[p])) top[p]=v;
        }
        printf("  step %2d :",step);
        for(int k=0;k<5;++k) printf("  %s(%.2f)", tokname(top[k]), lg_at(lg,top[k]));
        printf("%c",10);
        int nx=top[0];
        if(nx>ASTER_BYTE_MAX){ printf("  -> model picked control token %d at step %d%c",nx,step,10); break; }
        seq[n++]=(uint16_t)nx;
    }
    printf("generated: ");
    for(int i=16;i<n;++i) putchar(seq[i]<256?seq[i]:'.');
    printf("%c",10);
    return 0;
}
