
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
#define VECTOR_SIZE 256
#define ARRAY_SIZE 512
#define PE_SIZE 256
// int pdot[PE_SIZE][ARRAY_SIZE][VECTOR_SIZE]; 
// is too large, we had to allocate using malloc.

void clear_vector(int* v)
{
  memset(v, 0, sizeof(int) * VECTOR_SIZE);
}

void clear_array(int* a)
{
  memset(a, 0, sizeof(int) * ARRAY_SIZE);
}

int** array2D()
{
  int** array = (int**) malloc(sizeof(int*) * PE_SIZE);
  for (int i=0; i<PE_SIZE; i++) {
    array[i] = (int*) malloc(sizeof(int) * ARRAY_SIZE);
    clear_array(array[i]);
  }
  return array;
}

int*** array3D()
{
  int*** array = (int***) malloc(sizeof(int**) * PE_SIZE);
  for (int i=0; i<PE_SIZE; i++) {
    array[i] = (int**) malloc(sizeof(int*) * ARRAY_SIZE);
    for (int j=0; j<ARRAY_SIZE; j++) {
      array[i][j] = (int*) malloc(sizeof(int) * VECTOR_SIZE);
      clear_vector(array[i][j]);
    }
  }
  return array;
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
#define METRIC_STALL 12

int pim(int* x, int* w, int* y, int* lut_var, int* lut_rpr, int* metrics, int adc, int skip, int R, int D, int C, int NWL, int NBL, int WL, int BL)
{
  // x = nrow, nwl, wl, xb
  // f = nwl, wl, nbl, bl
  // y = nrow, ncol
  
  // our arrays are sized for 128. need to increase.
  // printf("%d %d %d\n", D, NWL * NBL, BL);
  assert ((D >= 1) && (NWL >= 1) && (NWL >= 1) && (BL >= 1));
  assert (D <= PE_SIZE);
  assert ((NWL * NBL) <= ARRAY_SIZE);
  assert (BL <= VECTOR_SIZE);
  
  int done = 0;
    
  int dup_done[PE_SIZE];
  int** array_done = array2D();

  int** wl_ptr = array2D();
  int** wl_sum = array2D();
  int** wl_total = array2D();
  
  int r[PE_SIZE]; 
  int** col = array2D();
  int** xb = array2D();
  
  int*** pdot = array3D();
  int*** pdot_sum = array3D();
  int*** sat = array3D();

  for (int d=0; d<D; d++) { 
    clear_array(wl_ptr[d]);
    clear_array(wl_total[d]);
    clear_array(col[d]);
    clear_array(xb[d]);
    clear_array(array_done[d]);
    
    r[d] = d;
    dup_done[d] = 0;
  }
  
  int next_r = D;
  
  while (!done) {
    // array_sync = 0;
    // clear_array(array_done);
    // clear_array(wl_ptr);
    // clear_array(xb);

    metrics[METRIC_CYCLE] += 1;
    // if there are more duplicates than rows, then I believe we hit this assert.
    assert (metrics[METRIC_CYCLE] < 1000000);

    for (int d=0; d<D; d++) { 
      for (int wl=0; wl<NWL; wl++) {
        for (int bl=0; bl<NBL; bl++) {

          int array = wl * NBL + bl;
          if (array_done[d][array]) {
            metrics[METRIC_STALL] += 1;
            continue;
          }
          
          /////////////////////////////////////

          int rpr_addr;
          if (BL >= C) {
            int x_addr = (xb[d][array] * 8);
            int w_addr = ((bl + 1) * (BL / C)) - 1;
            // for dense:
            w_addr = min(w_addr, 7);
            rpr_addr = x_addr + w_addr;
          }
          else {
            rpr_addr = (xb[d][array] * 8) + (bl / (C / BL)); 
          }
          
          int wbit = ((bl + 1) * (BL / C)) - 1;
          
          if (!((rpr_addr >= 0) && (rpr_addr < 64))) {
            printf("xb: %d bl: %d BL: %d C: %d: rpr_addr: %d\n", xb[d][array], bl, BL, C, rpr_addr);
            assert ((rpr_addr >= 0) && (rpr_addr < 64));
          }
          int rpr = lut_rpr[rpr_addr];
          assert (rpr >= 1);
          
          /////////////////////////////////////
          
          int rows = min(rpr, WL - wl_ptr[d][array]);

          clear_vector(pdot[d][array]);
          wl_sum[d][array] = 0;

          if (skip) {
            while ((wl_ptr[d][array] < WL) && (wl_sum[d][array] + x[(r[d] * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr[d][array] * 8) + xb[d][array]] <= rows)) {
              if (x[(r[d] * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr[d][array] * 8) + xb[d][array]]) {
                wl_sum[d][array] += 1;
                for (int adc_ptr=0; adc_ptr<BL; adc_ptr+=8) {
                  int bl_ptr = adc_ptr + col[d][array];
                  pdot[d][array][bl_ptr] += w[(wl * WL * NBL * BL) + (wl_ptr[d][array] * NBL * BL) + (bl * BL) + bl_ptr];
                }
              }
              wl_ptr[d][array] += 1;
            }
          }
          else {
            int start = wl_ptr[d][array];
            while ((wl_ptr[d][array] < WL) && (wl_ptr[d][array] < (start + adc))) {
              if (x[(r[d] * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr[d][array] * 8) + xb[d][array]]) {
                wl_sum[d][array] += 1;
                for (int adc_ptr=0; adc_ptr<BL; adc_ptr+=8) {
                  int bl_ptr = adc_ptr + col[d][array];
                  pdot[d][array][bl_ptr] += w[(wl * WL * NBL * BL) + (wl_ptr[d][array] * NBL * BL) + (bl * BL) + bl_ptr];
                }
              }
              wl_ptr[d][array] += 1;
            }
          }
          if (wl_sum[d][array] >= adc) {
            wl_total[d][array] += wl_sum[d][array];
          }
          
          /////////////////////////////////////

          for (int adc_ptr=0; adc_ptr<BL; adc_ptr+=8) {
            int bl_ptr = adc_ptr + col[d][array];
            int c = (bl_ptr + bl * BL) % C;
            int wb = (bl_ptr + bl * BL) / C;

            if (wb == 0) {
              y[r[d] * C + c] -= ((wl_sum[d][array] * 128) << xb[d][array]);
            }
            
            int key = rand() % 1000;
            int var_addr = pdot[d][array][bl_ptr] * 1000 + key;
            int var = lut_var[var_addr];

            if (!((var >= -3) && (var <= 3))) {
              printf("%d\n", var);
              assert ((var >= -3) && (var <= 3));
            }
            
            metrics[METRIC_RON] += pdot[d][array][bl_ptr];
            metrics[METRIC_ROFF] += rows - pdot[d][array][bl_ptr];

            pdot[d][array][bl_ptr] = min(max(pdot[d][array][bl_ptr] + var, 0), adc);
            y[r[d] * C + c] += (pdot[d][array][bl_ptr] << (wb + xb[d][array]));
            
            if (wl_sum[d][array] >= adc) {
              sat[d][array][bl_ptr] += (pdot[d][array][bl_ptr] == adc);
              pdot_sum[d][array][bl_ptr] += pdot[d][array][bl_ptr];
            }
          }

          int comps = min(wl_sum[d][array], min(rows, adc) - 1);
          //if (!((comps >= 0) && (comps < adc))) {
          //  printf("comps: %d wl_sum: %d rows: %d adc: %d\n", comps, wl_sum, rows, adc);
          //  assert((comps >= 0) && (comps < adc));
          //}
          metrics[comps] += BL;
          // assert(metrics[comps] < 1e9);
          metrics[METRIC_WL] += wl_sum[d][array];

          if (wl_ptr[d][array] == WL) {
            for (int adc_ptr=0; adc_ptr<BL; adc_ptr+=8) {
              int bl_ptr = adc_ptr + col[d][array];
              int c = (bl_ptr + bl * BL) % C;
              int wb = (bl_ptr + bl * BL) / C;

              if (wl_total[d][array]) {
                float p = ((float) pdot_sum[d][array][bl_ptr]) / ((float) wl_total[d][array]);
                p = min(max(p, 0.), 1.);
                int e = sat_error(p, adc, rpr);
                y[r[d] * C + c] -= ((sat[d][array][bl_ptr] * e) << (wb + xb[d][array]));
                
                sat[d][array][bl_ptr] = 0;
                pdot_sum[d][array][bl_ptr] = 0;
              }
            }
          }

          if (wl_ptr[d][array] == WL) {
            wl_ptr[d][array] = 0;
            wl_total[d][array] = 0;
            
            if (col[d][array] == (8 - 1)) {
              col[d][array] = 0;
          
              if (xb[d][array] == (8 - 1)) {
                xb[d][array] = 0;
                array_done[d][array] = 1;
                
                int array_sync = 1;
                for (int a=0; a<NWL * NBL; a++) {
                  array_sync = array_sync & array_done[d][a];
                }
                
                if (array_sync) {
                  if (next_r < R) {
                    r[d] = next_r;
                    next_r++;
                    clear_array(array_done[d]);
                  }
                  else {
                    dup_done[d] = 1;
                    
                    int dup_sync = 1;
                    for (int a=0; a<D; a++) {
                      dup_sync = dup_sync & dup_done[a];
                    }
                    
                    done = dup_sync;                  
                  }
                }
              }
              else {
                xb[d][array] += 1;
              }
            }
            else {
              col[d][array] += 1;
            }
          }
          else {
            assert (wl_ptr[d][array] < WL);
          }
          
        } // for (int bl=0; bl<NBL; bl++) {
      } // for (int wl=0; wl<NWL; wl++) {
    } // for (int d=0; d<D; d++) { 
  } // while (!done) {
    
  // printf("%d: %d %d\n", NWL * NBL, metrics[METRIC_CYCLE], metrics[METRIC_STALL]);
  
  return metrics[METRIC_CYCLE];  
}





























