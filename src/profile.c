
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

int profile(int* x, int* w, int* y, long* count, int max_rpr, int R, int C, int NWL, int NBL, int WL, int BL)
{
  int pdot[VECTOR_SIZE];

  // x = nrow, nwl, wl, xb
  // f = nwl, wl, nbl, bl
  // y = nrow, ncol
  
  for (int rpr=1; rpr<=max_rpr; rpr++) {
    printf("%d\n", rpr);
    
    for (int r=0; r<R; r++) {
      for (int wl=0; wl<NWL; wl++) {
        for (int bl=0; bl<NBL; bl++) {
          for (int xb=0; xb<8; xb++) {
            
            int wl_total = 0;
            int wl_ptr = 0;
            while (wl_ptr < WL) {
              
              clear(pdot);
              int wl_sum = 0;

              while ((wl_ptr < WL) && (wl_sum + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb] <= rpr)) {
                if (x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb]) {
                  wl_sum += 1;
                  for (int bl_ptr=0; bl_ptr<BL; bl_ptr++) {
                    pdot[bl_ptr] += w[(wl * WL * NBL * BL) + (wl_ptr * NBL * BL) + (bl * BL) + bl_ptr];
                  }
                }
                wl_ptr += 1;
              }

              for (int bl_ptr=0; bl_ptr<BL; bl_ptr++) {
                int val = pdot[bl_ptr];
                count[rpr * (max_rpr+1) + val] += 1;
              }

            } // while (wl_ptr < wl) {                  
          } // for (int xb=0; xb<8; xb++) {
        } // for (int bl=0; bl<BL; bl++) {
      } // for (int wl=0; wl<WL; wl++) {
    } // for (int r=0; r<R; r++) {
  } // for (int rpr=0; rpr<max_rpr; rpr++) {
    
  return 1;  
}

































