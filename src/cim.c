

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

#define VECTOR_SIZE 65536
#define WORD_SIZE 64

//////////////////////////////////////////////

void clear(int* v)
{
  memset(v, 0, sizeof(int) * VECTOR_SIZE);
}

//////////////////////////////////////////////

int cim(int8_t* x, int8_t* w, uint8_t* cim_ref, uint8_t* cim_var, uint8_t* count, uint8_t* rpr_table, float* var_table, int R, int NWL, int WL, int BL) {
  int pdot[VECTOR_SIZE];

  for (int r=0; r<R; r++) {
    for (int wl=0; wl<NWL; wl++) {
      for (int xb=0; xb<8; xb++) {
        for (int wb=0; wb<8; wb++) {

          int wl_ptr = 0;
          int wl_itr = 0;
          while (wl_ptr < WL) {
            int wl_sum = 0;
            clear(pdot);

            int rpr = rpr_table[xb * 8 + wb];
            assert (rpr >= 1);
            while ((wl_ptr < WL) && (wl_sum + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb] <= rpr)) {
              if (x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb]) {
                wl_sum += 1;
                for (int bl_ptr=0; bl_ptr<BL; bl_ptr+=8) {
                  int bl = bl_ptr + wb;
                  pdot[bl] += w[(wl * WL * BL) + (wl_ptr * BL) + bl];
                }
              }
              wl_ptr += 1;
            }

            for (int bl_ptr=0; bl_ptr<BL; bl_ptr+=8) {
              int bl = bl_ptr + wb;

              int row_addr = r * NWL * 8 * BL * WORD_SIZE;
              int wl_addr =       wl * 8 * BL * WORD_SIZE;
              int xb_addr =           xb * BL * WORD_SIZE;
              int bl_addr =                bl * WORD_SIZE;

              int y_addr = row_addr + wl_addr + xb_addr + bl_addr + wl_itr;

              int key = rand() % 1001;
              int var_addr = pdot[bl] * 1001 + key;
              float var = var_table[var_addr];
              float pdot_var = pdot[bl] + var;

              cim_ref[y_addr] = pdot[bl];
              if ((pdot_var > 0.20) && (pdot_var < 1.00)) cim_var[y_addr] = 1;
              else                                        cim_var[y_addr] = min(max((int) round(pdot_var), 0), min(8, rpr));

              // cim_ref[y_addr] += 1;
              // cim_var[y_addr] += 1;
            }

            int row_addr = r * NWL * 8 * WORD_SIZE;
            int wl_addr =       wl * 8 * WORD_SIZE;
            int xb_addr =           xb * WORD_SIZE;
            int count_addr = row_addr + wl_addr + xb_addr + wl_itr;
            count[count_addr] = wl_sum;

            assert (wl_itr < WORD_SIZE);
            wl_itr += 1;
          } // while (wl_ptr < wl) {
        } // for (int bl=0; bl<BL; bl++) {
      } // for (int xb=0; xb<8; xb++) {
    } // for (int wl=0; wl<WL; wl++) {
  } // for (int r=0; r<R; r++) {
  return 1;  
}

































