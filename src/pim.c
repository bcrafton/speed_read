
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

void clear(int* v)
{
    memset(v, 0, sizeof(int) * VECTOR_SIZE);
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

int sat_error(float p, int adc, int rpr)
{
  float e = 0.;
  for (int s=adc; s<rpr; s++) {
    float bin = binomial_pmf(s, rpr, p);
    e += bin * (adc - s);
  }
  return e;
}

int pim(int* x, int* w, int* y, int* lut_var, int* lut_rpr, int adc, int R, int C, int NWL, int NBL, int WL, int BL)
{
  int pdot[VECTOR_SIZE];
  int pdot_sum[VECTOR_SIZE];
  int sat[VECTOR_SIZE];

  // x = nrow, nwl, wl, xb
  // f = nwl, wl, nbl, bl
  // y = nrow, ncol
  int psum = 0;
  
  for (int r=0; r<R; r++) {
    for (int wl=0; wl<NWL; wl++) {
      for (int bl=0; bl<NBL; bl++) {
        for (int xb=0; xb<8; xb++) {

          // TODO: dont want to clip -> get this function right.
          int rpr_addr = (xb * 8) + ((bl + 1) * (BL / C)) - 1;
          if (rpr_addr == -1) rpr_addr = 0;

          if (!((rpr_addr >= 0) && (rpr_addr < 64))) {
            printf("%d %d %d %d: %d\n", xb, bl, BL, C, rpr_addr);
            assert ((rpr_addr >= 0) && (rpr_addr < 64));
          }
          int rpr = lut_rpr[rpr_addr];
          
          clear(sat);
          clear(pdot_sum);
          int wl_total = 0;
          int wl_ptr = 0;
          while (wl_ptr < WL) {
          
            clear(pdot);
            int wl_sum = 0;
            while ((wl_ptr < WL) && (wl_sum + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb] <= rpr)) { // adc -> rpr
              if (x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb]) {
                wl_sum += 1;
                wl_total += 1;
                for (int bl_ptr=0; bl_ptr<BL; bl_ptr++) {
                  pdot[bl_ptr] += w[(wl * WL * NBL * BL) + (wl_ptr * NBL * BL) + (bl * BL) + bl_ptr];
                }
              }
              wl_ptr += 1;
            }
            psum += 1;
            
            for (int bl_ptr=0; bl_ptr<BL; bl_ptr++) {
              int c = (bl_ptr + bl * BL) % C;
              int wb = (bl_ptr + bl * BL) / C;
              // comment me out for speed.
              // if ((wl_ptr == 0) && (wl == 0) && (xb == 0) && (wb == 0)) { assert(y[r * C + c] == 0); }

              if (wb == 0) {
                y[r * C + c] -= ((wl_sum * 128) << xb);
              }
              
              int key = rand() % 1000;
              int var_addr = pdot[bl_ptr] * 1000 + key;
              int var = lut_var[var_addr];

              if (!((var >= -3) && (var <= 3))) {
                printf("%d\n", var);
                assert ((var >= -3) && (var <= 3));
              }              

              pdot[bl_ptr] = min(max(pdot[bl_ptr] + var, 0), adc);
              y[r * C + c] += (pdot[bl_ptr] << (wb + xb));
              
              // ordering matters here.
              // do not put sat/pdot_sum before any pdot changes.
              sat[bl_ptr] += (pdot[bl_ptr] == adc); 
              pdot_sum[bl_ptr] += pdot[bl_ptr];
            }

          } // while (wl_ptr < wl) {

          for (int bl_ptr=0; bl_ptr<BL; bl_ptr++) {
            int c = (bl_ptr + bl * BL) % C;
            int wb = (bl_ptr + bl * BL) / C;
            if (wl_total) {
              float p = ((float) pdot_sum[bl_ptr]) / ((float) wl_total);
              p = min(max(p, 0.), 1.);
              assert (p <= 1.);
              int e = (int) round(sat_error(p, adc, rpr));
              assert ((e >= -1) && (e <= 0));
              // assert(sat[bl_ptr] * e == 0);
              // if (sat[bl_ptr] * e) printf ("%d\n", sat[bl_ptr] * e);
              // printf("(%d %d: %f) (%d %d: %d %d)\n", pdot_sum[bl_ptr], wl_total, p, adc, rpr, e, sat[bl_ptr]);
              y[r * C + c] -= ((sat[bl_ptr] * e) << (wb + xb));
            }
          }
        
        } // for (int xb=0; xb<8; xb++) {
      } // for (int bl=0; bl<BL; bl++) {
    } // for (int wl=0; wl<WL; wl++) {
  } // for (int r=0; r<R; r++) {
  
  return psum;  
}

































