
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
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

int pim_kernel(int* x, int* w, int wl, int bl, int* y)
{
    int ncol = bl / 8;

    int psum = 0;
    int wl_ptr = 0;

    while (wl_ptr < wl) {
        int wl_sum = 0;
        clear_pdot();
        
        while ((wl_ptr < wl) && (wl_sum + x[wl_ptr] <= 8)) {
            if (x[wl_ptr]) {
                wl_sum += 1;
                for (int bl_ptr=0; bl_ptr<bl; bl_ptr++) {
                    pdot[bl_ptr] += w[wl_ptr * bl + bl_ptr];
                }
            }
            wl_ptr += 1;
        }
        psum += 1;

        for (int col=0; col<ncol; col++) {
            for (int wb=0; wb<8; wb++) {
                y[col] += (pdot[col * 8 + wb] << wb);
            }
            y[col] -= wl_sum * 128;
        }
    }
    return psum;
}

int conv(int* x, int* f, int* y, int S, int X, int Y, int K, int C, int N)
{
  for (int yh=0; yh<Y; yh++) {
    for (int yw=0; yw<Y; yw++) {
      
      for (int kh=0; kh<K; kh++) {
        for (int kw=0; kw<K; kw++) {
        
          for (int c=0; c<C; c++) {
            for (int n=0; n<N; n++) {
              int x_addr = ((yh + kh) * X * C) + ((yw + kw) * C) + c;
              int f_addr = (kh * K * C * N) + (kw * C * N) + (c * N) + n;
              int y_addr = (yh * Y * N) + (yw * N) + n;
              y[y_addr] += x[x_addr] * f[f_addr];
            }
          }
        
        }
      }
      
    }
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
        
        } // for (int xb=0; xb<8; xb++) {
      } // for (int bl=0; bl<BL; bl++) {
    } // for (int wl=0; wl<WL; wl++) {
  } // for (int r=0; r<R; r++) {
  
  return psum;  
}

































