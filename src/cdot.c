
#include <stdio.h>

// make sure (bl <= 1024), malloc would be too slow.
// if we just pick a size large enough we will be okay
#define VECTOR_SIZE 1024
int pdot[VECTOR_SIZE]; 

int pim_kernel(int* x, int* w, int wl, int bl, int* y)
{

    int ncol = bl / 8;

    int psum = 0;
    int wl_ptr = 0;
        
    for (int bl_ptr=0; bl_ptr<bl; bl_ptr++) { pdot[bl_ptr] = 0; }
    for (int col=0; col<ncol; col++)        { y[col] = 0; }
    
    while (wl_ptr < wl) {
        int wl_sum = 0;
        for (int bl_ptr=0; bl_ptr<bl; bl_ptr++) { pdot[bl_ptr] = 0; }
        
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








