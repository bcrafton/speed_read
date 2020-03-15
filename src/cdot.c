
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <assert.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

//////////////////////////////////////////////

// make sure (bl <= 1024), malloc would be too slow.
// if we just pick a size large enough we will be okay
#define VECTOR_SIZE 1024
int pdot[VECTOR_SIZE];
int pdot_sum[VECTOR_SIZE];
int sat[VECTOR_SIZE];
int sat_err[VECTOR_SIZE];

void clear_pdot()
{
    memset(pdot, 0, sizeof(int) * VECTOR_SIZE);
}

void clear_pdot_sum()
{
    memset(pdot_sum, 0, sizeof(int) * VECTOR_SIZE);
}

void clear_sat()
{
    memset(sat, 0, sizeof(int) * VECTOR_SIZE);
}

//////////////////////////////////////////////

int factorial(int n)
{
  int fact = 1;
  for (int i=1; i<(n+1); i++) {
    fact = fact * i;
  }
  return fact;
}

float binomial_pmf(int k, int n, float p)
{
  int nCk = factorial(n) / (factorial(k) * factorial(n - k));
  float success = pow(p, k);
  float fail = pow(1 - p, n - k);
  return nCk * success * fail;
}

void calc_sat_error(float* p, int* sat, int* e, int len, int adc, int rpr)
{
  for (int i=0; i<len; i++) {
    float mu;
    for (int s=adc; s<rpr; s++) {
      float bin = binomial_pmf(s, rpr, p[i]);
      mu += bin * (adc - s);
    }
    e[i] = sat[i] * round(mu);
  }
}

int pim(int* x, int* w, int* y, int* lut, int R, int C, int NWL, int NBL, int WL, int BL)
{
  // x = nrow, nwl, wl, xb
  // f = nwl, wl, nbl, bl
  // y = nrow, ncol
  int psum = 0;
  
  for (int r=0; r<R; r++) {
    for (int wl=0; wl<NWL; wl++) {
      for (int bl=0; bl<NBL; bl++) {
        for (int xb=0; xb<8; xb++) {
        
          clear_sat();
          clear_pdot_sum();
          int wl_ptr = 0;
          while (wl_ptr < WL) {
          
            clear_pdot();
            int wl_sum = 0;
            while ((wl_ptr < WL) && (wl_sum + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb] <= 8)) {
              if (x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb]) {
                wl_sum += 1;
                for (int bl_ptr=0; bl_ptr<BL; bl_ptr++) {
                  pdot[bl_ptr] += w[(wl * WL * NBL * BL) + (wl_ptr * NBL * BL) + (bl * BL) + bl_ptr];
                }
              }
              wl_ptr += 1;
            }
            psum += 1;
            
            for (int bl_ptr=0; bl_ptr<BL; bl_ptr++) {
              // ordering matters here.
              // do not put sat/pdot_sum behind any pdot changes.
              sat[bl_ptr] += (pdot[bl_ptr] == 8); // 8 = adc
              pdot_sum[bl_ptr] += pdot_sum[bl_ptr];

              int c = (bl_ptr + bl * BL) % C;
              int wb = (bl_ptr + bl * BL) / C;
              // comment me out for speed.
              // if ((wl_ptr == 0) && (wl == 0) && (xb == 0) && (wb == 0)) { assert(y[r * C + c] == 0); }

              if (wb == 0) {
                y[r * C + c] -= ((wl_sum * 128) << xb);
              }
              
              int key = rand() % 1000;
              int var_addr = pdot[bl_ptr] * 1000 + key;
              int var = lut[var_addr];
              assert ((var > -3) && (var < 3));
              // add and clip var to pdot.
              y[r * C + c] += ((pdot[bl_ptr] + var) << (wb + xb));
            }

          } // while (wl_ptr < wl) {
          
          /*
          calc_sat_error();
          for (int c=0; c<C; c++) {
            y[r * C + c] -= sat_error();
          }
          */
        
        } // for (int xb=0; xb<8; xb++) {
      } // for (int bl=0; bl<BL; bl++) {
    } // for (int wl=0; wl<WL; wl++) {
  } // for (int r=0; r<R; r++) {
  
  return psum;  
}

































