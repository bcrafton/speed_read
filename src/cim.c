

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

// typedef char int8_t; // already defined ? 
typedef unsigned char uint8_t;

//////////////////////////////////////////////

// make sure (bl <= 1024), malloc would be too slow.
// if we just pick a size large enough we will be okay
#define VECTOR_SIZE 1024

void clear(int* v)
{
  memset(v, 0, sizeof(int) * VECTOR_SIZE);
}

//////////////////////////////////////////////

int cim(int8_t* x, int8_t* w, uint8_t* y, uint8_t* rpr_table, int R, int NWL, int WL, int BL) {
  for (int r=0; r<R; r++) {
    for (int wl=0; wl<NWL; wl++) {
      for (int xb=0; xb<8; xb++) {
        for (int bl=0; bl<BL; bl++) {

          int wl_ptr = 0;
          int wl_itr = 0;
          while (wl_ptr < WL) {
            int wl_sum = 0;
            int pdot = 0;

            int rpr = rpr_table[xb * 8 + (bl % 8)];
            // assert(rpr >= 1);
            while ((wl_ptr < WL) && (wl_sum + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb] <= rpr)) {
              if (x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb]) {
                wl_sum += 1;
                pdot += w[(wl * WL * BL) + (wl_ptr * BL) + bl];
              }
              wl_ptr += 1;
            }

            int row_addr = r * NWL * 8 * BL * 17;
            int wl_addr =       wl * 8 * BL * 17;
            int xb_addr =           xb * BL * 17;
            int bl_addr =                bl * 17;

            if (rpr <= 16) {
              int y_addr = row_addr + wl_addr + xb_addr + bl_addr + pdot;
              y[y_addr] += 1;
            }
            else {
              int y_addr = row_addr + wl_addr + xb_addr + bl_addr + wl_itr;
              y[y_addr] = pdot;
            }

            wl_itr += 1;

          } // while (wl_ptr < wl) {
        } // for (int bl=0; bl<BL; bl++) {
      } // for (int xb=0; xb<8; xb++) {
    } // for (int wl=0; wl<WL; wl++) {
  } // for (int r=0; r<R; r++) {
  return 1;  
}

































