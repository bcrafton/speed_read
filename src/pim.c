
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
#define ARRAY_SIZE 128

void clear_vector(int* v)
{
  memset(v, 0, sizeof(int) * VECTOR_SIZE);
}

void clear_array(int* a)
{
  memset(a, 0, sizeof(int) * ARRAY_SIZE);
}

//////////////////////////////////////////////

long unsigned int factorial(int n)
{
  long unsigned int fact = 1;
  for (int i=1; i<(n+1); i++) {
    fact = fact * i;
  }
  return fact;
}

long unsigned int nChoosek(int n, int k)
{
  long unsigned int t = factorial(n);
  long unsigned int b = factorial(k) * factorial(n - k);
  long unsigned int nCk = t / b;
  assert (nCk > 0);
  return nCk;
}

float binomial_pmf(int k, int n, float p)
{
  long unsigned int nCk = nChoosek(n, k);
  float success = pow(p, k);
  float fail = pow(1 - p, n - k);
  float pmf = nCk * success * fail;
  assert (pmf >= 0.);
  return pmf;
}

int sat_error(float p, int adc, int rpr)
{
  if (rpr <= adc) {
    return 0;
  }
  float e = 0.;
  float bin_sum = 0.;
  for (int s=adc; s<(rpr+1); s++) {
    float bin = binomial_pmf(s, rpr, p);
    bin_sum += bin;
    e += bin * (adc - s);
  }
  assert (bin_sum >= 0.);
  if (bin_sum > 0.) {
    e /= bin_sum;
  }
  assert (e <= 0);
  return e;
}

/*
metrics
------
adc.1
adc.2
adc.3
adc.4
adc.5
adc.6
adc.7
adc.8
cycle
ron
roff
wl
*/

#define METRIC_CYCLE  8
#define METRIC_RON    9
#define METRIC_ROFF  10
#define METRIC_WL    11

int pim(int* x, int* w, int* y, int* lut_var, int* lut_rpr, int* metrics, int adc, int skip, int R, int C, int NWL, int NBL, int WL, int BL)
{
  printf("%d\n", NWL * NBL);

  // x = nrow, nwl, wl, xb
  // f = nwl, wl, nbl, bl
  // y = nrow, ncol
  
  int cycles = 0;
  
  int array_sync = 0;
  int array_done[ARRAY_SIZE];

  int wl_ptr[ARRAY_SIZE]; // NWL * NBL
  int wl_sum[ARRAY_SIZE]; // NWL * NBL
  // int wl_total[ARRAY_SIZE]; // NWL * NBL
  
  int xb[ARRAY_SIZE]; // NWL * NBL
    
  // int r[ARRAY_SIZE]; // NWL * NBL // this will be needed at duplicate level.
  
  int pdot[ARRAY_SIZE][VECTOR_SIZE];
  // int pdot_sum[ARRAY_SIZE][VECTOR_SIZE];
  // int sat[ARRAY_SIZE][VECTOR_SIZE];
  
  for (int r=0; r<R; r++) {
    array_sync = 0;
    clear_array(array_done);
    
    clear_array(wl_ptr);
    clear_array(xb);
    
    while (!array_sync) {
      cycles += 1;
      for (int wl=0; wl<NWL; wl++) {
        for (int bl=0; bl<NBL; bl++) {

          int array = wl * NBL + bl;
          if (array_done[array]) { 
            // printf("array %d done\n", array);
            continue;
          }
          else {
            // printf("array %d not done\n", array);
          }
          
          clear_vector(pdot[array]);
          wl_sum[array] = 0;

          /////////////////////////////////////

          int rpr_addr;
          if (BL >= C) {
            int x_addr = (xb[array] * 8);
            int w_addr = ((bl + 1) * (BL / C)) - 1;
            // for dense:
            w_addr = min(w_addr, 7);
            rpr_addr = x_addr + w_addr;
          }
          else {
            rpr_addr = (xb[array] * 8) + (bl / (C / BL)); 
          }
          
          if (!((rpr_addr >= 0) && (rpr_addr < 64))) {
            printf("xb: %d bl: %d BL: %d C: %d: rpr_addr: %d\n", xb[array], bl, BL, C, rpr_addr);
            assert ((rpr_addr >= 0) && (rpr_addr < 64));
          }
          int rpr = lut_rpr[rpr_addr];
          assert (rpr >= 1);
          
          /////////////////////////////////////
          
          int rows = min(rpr, WL - wl_ptr[array]);
            
          // printf("%d %d %d\n", wl_ptr[array], (wl_ptr[array] < WL), (wl_sum[array] + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr[array] * 8) + xb[array]] <= rows));
            
          if (skip) {
            while ((wl_ptr[array] < WL) && (wl_sum[array] + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr[array] * 8) + xb[array]] <= rows)) {
              if (x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr[array] * 8) + xb[array]]) {
                wl_sum[array] += 1;
                for (int bl_ptr=0; bl_ptr<BL; bl_ptr++) {
                  pdot[array][bl_ptr] += w[(wl * WL * NBL * BL) + (wl_ptr[array] * NBL * BL) + (bl * BL) + bl_ptr];
                }
              }
              wl_ptr[array] += 1;
            }
          }
          else {
            assert(0);
          }
          
          /////////////////////////////////////

          for (int bl_ptr=0; bl_ptr<BL; bl_ptr++) {
            int c = (bl_ptr + bl * BL) % C;
            int wb = (bl_ptr + bl * BL) / C;
            // comment me out for speed.
            // if ((wl_ptr == 0) && (wl == 0) && (xb == 0) && (wb == 0)) { assert(y[r * C + c] == 0); }

            if (wb == 0) {
              y[r * C + c] -= ((wl_sum[array] * 128) << xb[array]);
            }
            
            int key = rand() % 1000;
            int var_addr = pdot[array][bl_ptr] * 1000 + key;
            int var = lut_var[var_addr];

            if (!((var >= -3) && (var <= 3))) {
              printf("%d\n", var);
              assert ((var >= -3) && (var <= 3));
            }

            pdot[array][bl_ptr] = min(max(pdot[array][bl_ptr] + var, 0), adc);
            y[r * C + c] += (pdot[array][bl_ptr] << (wb + xb[array]));
          }
                    
          if (wl_ptr[array] == WL) {            
            wl_ptr[array] = 0;
            
            if (xb[array] == (8 - 1)) {
              xb[array] = 0;
              array_done[array] = 1;
              
              array_sync = 1;
              for (int a=0; a<NWL * NBL; a++) {
                array_sync = array_sync & array_done[a];
              }
            }
            else {
              xb[array] += 1;
            }
          }
          else {
            assert (wl_ptr[array] < WL);
          }
          
        } // for (int bl=0; bl<NBL; bl++) {
      } // for (int wl=0; wl<NWL; wl++) {
      assert (cycles < 100000);
    } // while (!done) {
  } // for (int r=0; r<R; r++) {
  
  metrics[METRIC_CYCLE] = cycles;
  
  return cycles;  
}





























