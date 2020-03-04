/* Written by Sam Neymotin (samuel.neymotin@nki.rfmh.org)

this is used for calculating phase locking value in phaselock.py
via the ctypes interface

compilation:
 gcc -Wall -fPIC -c phslock.c 
 gcc -shared -o phslock.so phslock.o

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <time.h>

//shuffle array of doubles
void dshuffle (double* x,int nx) {
  srand(time(NULL));
  int n,k; double temp;
  for (n=nx;n>1;) {
    n--;
    k = rand() % nx; 
    temp = x[n];
    x[n] = x[k];
    x[k] = temp;
  }  
}

double phslockv (double *lphsa,double *lphsb, int num) {
  double complex phslv = 0.0;
  int i;
  for(i=0;i<num;i++) {    
    phslv += cexp(I * (lphsa[i] - lphsb[i]));
  }
  phslv /= (double complex)num;
  return cabs(phslv);
}

double phslockvshuf (double *lphsa,double *lphsb, int sz, int nshuf) {
  if (nshuf <= 0) return phslockv(lphsa,lphsb,sz);
  int i;
  double* tmpb = (double*)malloc(sizeof(double)*sz);
  double avgplvshuf = 0.0;
  double avgplvshuf2 = 0.0;
  double plvshuf = 0.0;
  double plv = phslockv(lphsa,lphsb,sz);
  memcpy(tmpb,lphsb,sizeof(double)*sz);
  for(i=0;i<nshuf;i++) {
    dshuffle(tmpb,sz); // shuffle
    plvshuf = phslockv(lphsa,tmpb,sz);
    avgplvshuf += plvshuf;
    avgplvshuf2 += plvshuf*plvshuf;
  }
  free(tmpb);
  avgplvshuf /= (double)nshuf; // average from shuffled
  avgplvshuf2/=(double)nshuf;
  avgplvshuf2 -= avgplvshuf*avgplvshuf;
  if(avgplvshuf2>0.) avgplvshuf2=sqrt(avgplvshuf2); // standard deviation
  printf("plv=%g,avgplvshuf=%g, avgplvshuf2=%g\n",plv,avgplvshuf,avgplvshuf2);
  if(avgplvshuf2>0.) {
    return (plv - avgplvshuf) / avgplvshuf2; // z-score
  } else {
    return plv - avgplvshuf; // difference of real plv from average shuffled plv
  }
}
