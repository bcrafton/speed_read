
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
#define VECTOR_SIZE 128 // number bl per array
#define ARRAY_SIZE 32 // 512 / 16 = 32
#define BLOCK_SIZE 2048 // number of blocks 

//////////////////////////////////////////////

void clear_vector(int* v)
{
  memset(v, 0, sizeof(int) * VECTOR_SIZE);
}

void clear_array(int* a)
{
  memset(a, 0, sizeof(int) * ARRAY_SIZE);
}

void clear_block(int* a)
{
  memset(a, 0, sizeof(int) * BLOCK_SIZE);
}

//////////////////////////////////////////////

void free3D(int*** array)
{
  for (int i=0; i<BLOCK_SIZE; i++) {
    for (int j=0; j<ARRAY_SIZE; j++) {
      free(array[i][j]);
    }
    free(array[i]);
  }
  free(array);
}

//////////////////////////////////////////////

int* array1D()
{
  int* array = (int*) malloc(sizeof(int) * BLOCK_SIZE);
  clear_block(array);
  return array;
}

int** array2D()
{
  int** array = (int**) malloc(sizeof(int*) * BLOCK_SIZE);
  for (int i=0; i<BLOCK_SIZE; i++) {
    array[i] = (int*) malloc(sizeof(int) * ARRAY_SIZE);
    clear_array(array[i]);
  }
  return array;
}

int*** array3D()
{
  int*** array = (int***) malloc(sizeof(int**) * BLOCK_SIZE);
  for (int i=0; i<BLOCK_SIZE; i++) {
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
#define METRIC_BLOCK_CYCLE 13

int pim(int* x, int* w, int* y, float* lut_var, int* lut_rpr, int* metrics, int* block_map, int adc, int skip, int R, int B, int C, int NWL, int NBL, int WL, int BL)
{
  // x = nrow, nwl, wl, xb
  // f = nwl, wl, nbl, bl
  // y = nrow, ncol
  
  // our arrays are sized for 128. need to increase.
  // assert ((D >= 1) && (NWL >= 1) && (NBL >= 1) && (BL >= 1));
  assert ((NWL >= 1) && (NBL >= 1) && (BL >= 1));
  assert (NBL <= ARRAY_SIZE);
  assert (BL <= VECTOR_SIZE);
  assert (B <= BLOCK_SIZE);
  
  int done = 0;
    
  int* r = array1D(); 
  int* next_r = array1D(); 
    
  int* block_done = array1D();
  int* col = array1D();
  int* xb = array1D();

  int** wl_ptr = array2D();
  int** wl_sum = array2D();
  int** wl_total = array2D();
  
  int*** pdot = array3D();
  int*** pdot_sum = array3D();
  int*** sat = array3D();

  for (int block=0; block<B; block++) {
    int wl = block_map[block];
    assert (wl < NWL);
    r[block] = next_r[wl];
    next_r[wl]++;
  }  
  // next_r will have to be 2D.
  // all {row, wl}

  while (!done) {

    for (int i=0; i<B; i++) {
      for (int j=0; j<NBL; j++) {
        assert(wl_ptr[i][0] == wl_ptr[i][j]);
      }
    }

    metrics[METRIC_CYCLE] += 1;
    // if there are more duplicates than rows, then I believe we hit this assert.
    // assert (metrics[METRIC_CYCLE] < 500000);

    for (int block=0; block<B; block++) {
      
      int wl = block_map[block];
      assert (wl < NWL);

      // here is our issue.
      if (block_done[block]) {
        metrics[METRIC_STALL] += NBL;
        continue;
      }
      else { 
        metrics[METRIC_BLOCK_CYCLE + wl] += 1;
      }
      
      for (int bl=0; bl<NBL; bl++) {
        
        int x_addr = xb[block] * 8;
        int w_addr = col[block];
        int rpr_addr = x_addr + w_addr;

        if (!((rpr_addr >= 0) && (rpr_addr < 64))) {
          printf("xb: %d bl: %d BL: %d C: %d: rpr_addr: %d\n", xb[block], bl, BL, C, rpr_addr);
          assert ((rpr_addr >= 0) && (rpr_addr < 64));
        }
        int rpr = lut_rpr[rpr_addr];
        assert (rpr >= 1);
                
        /////////////////////////////////////
        
        int rows = min(rpr, WL - wl_ptr[block][bl]);

        clear_vector(pdot[block][bl]);
        wl_sum[block][bl] = 0;

        if (skip) {
          while ((wl_ptr[block][bl] < WL) && (wl_sum[block][bl] + x[(r[block] * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr[block][bl] * 8) + xb[block]] <= rows)) {
            if (x[(r[block] * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr[block][bl] * 8) + xb[block]]) {
              wl_sum[block][bl] += 1;
              for (int adc_ptr=0; adc_ptr<BL; adc_ptr+=8) {
                int bl_ptr = adc_ptr + col[block];
                pdot[block][bl][bl_ptr] += w[(wl * WL * NBL * BL) + (wl_ptr[block][bl] * NBL * BL) + (bl * BL) + bl_ptr];
              }
            }
            wl_ptr[block][bl] += 1;
          }
        }
        else {
          int start = wl_ptr[block][bl];
          while ((wl_ptr[block][bl] < WL) && (wl_ptr[block][bl] < (start + adc))) {
            if (x[(r[block] * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr[block][bl] * 8) + xb[block]]) {
              wl_sum[block][bl] += 1;
              for (int adc_ptr=0; adc_ptr<BL; adc_ptr+=8) {
                int bl_ptr = adc_ptr + col[block];
                pdot[block][bl][bl_ptr] += w[(wl * WL * NBL * BL) + (wl_ptr[block][bl] * NBL * BL) + (bl * BL) + bl_ptr];
              }
            }
            wl_ptr[block][bl] += 1;
          }
        }
        if (wl_sum[block][bl] >= adc) {
          wl_total[block][bl] += wl_sum[block][bl];
        }
        
        /////////////////////////////////////

        for (int adc_ptr=0; adc_ptr<BL; adc_ptr+=8) {
          int bl_ptr = adc_ptr + col[block];
          int c = (bl_ptr + bl * BL) / 8;
          int wb = col[block];

          if (wb == 0) {
            y[r[block] * C + c] -= ((wl_sum[block][bl] * 128) << xb[block]);
          }
          
          int key = rand() % 1000;
          int var_addr = pdot[block][bl][bl_ptr] * 1000 + key;
          float var = lut_var[var_addr];

          if (!((var >= -3) && (var <= 3))) {
            printf("%f\n", var);
            assert ((var >= -3) && (var <= 3));
          }
          
          metrics[METRIC_RON] += pdot[block][bl][bl_ptr];
          metrics[METRIC_ROFF] += rows - pdot[block][bl][bl_ptr];

          // pdot[block][bl][bl_ptr] = min(max(pdot[block][bl][bl_ptr] + var, 0), adc);
          float pdot_var = pdot[block][bl][bl_ptr] + var;
          pdot[block][bl][bl_ptr] = min(max((int) round(pdot_var), 0), adc);
          y[r[block] * C + c] += (pdot[block][bl][bl_ptr] << (wb + xb[block]));
          
          if (wl_sum[block][bl] >= adc) {
            sat[block][bl][bl_ptr] += (pdot[block][bl][bl_ptr] == adc);
            pdot_sum[block][bl][bl_ptr] += pdot[block][bl][bl_ptr];
          }
        }

        int comps = min(wl_sum[block][bl], min(rows, adc) - 1);
        //if (!((comps >= 0) && (comps < adc))) {
        //  printf("comps: %d wl_sum: %d rows: %d adc: %d\n", comps, wl_sum, rows, adc);
        //  assert((comps >= 0) && (comps < adc));
        //}
        metrics[comps] += BL;
        // assert(metrics[comps] < 1e9);
        metrics[METRIC_WL] += wl_sum[block][bl];

        if (wl_ptr[block][bl] == WL) {
          for (int adc_ptr=0; adc_ptr<BL; adc_ptr+=8) {
            int bl_ptr = adc_ptr + col[block];
            int c = (bl_ptr + bl * BL) / 8;
            int wb = col[block];

            if (wl_total[block][bl]) {
              float p = ((float) pdot_sum[block][bl][bl_ptr]) / ((float) wl_total[block][bl]);
              p = min(max(p, 0.), 1.);
              int e = sat_error(p, adc, rpr);
              y[r[block] * C + c] -= ((sat[block][bl][bl_ptr] * e) << (wb + xb[block]));
              
              sat[block][bl][bl_ptr] = 0;
              pdot_sum[block][bl][bl_ptr] = 0;
            }
          }
        }

      
        if (wl_ptr[block][bl] == WL) {
          wl_ptr[block][bl] = 0;
          wl_total[block][bl] = 0;

          if (bl == (NBL - 1)) {
            if (col[block] == (8 - 1)) {
              col[block] = 0;
          
              if (xb[block] == (8 - 1)) {
                xb[block] = 0;
                
                if (next_r[wl] < R) {
                  r[block] = next_r[wl];
                  next_r[wl]++;
                }
                else {
                  block_done[block] = 1;
                  
                  int block_sync = 1;
                  for (int i=0; i<B; i++) {
                    block_sync = block_sync & block_done[i];
                  }

                  done = block_sync;                  
                }
              }
              else {
                xb[block] += 1;
              }
            }
            else {
              col[block] += 1;
            }
          }
        }
        else {
          assert (wl_ptr[block][bl] < WL);
        }

      } // for (int bl=0; bl<NBL; bl++) {
    } // for (int b=0; b<B; b++) {
  } // while (!done) {

  free3D(pdot);
  free3D(pdot_sum);
  free3D(sat);

  return metrics[METRIC_CYCLE];  
}





























