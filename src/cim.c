

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <assert.h>
#include <random>
#include <vector>
using namespace std;

#define DLLEXPORT extern "C"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifndef clip
#define clip(a,b,c)         ((a > b) ? b : (a < c) ? c : a)
#endif

//////////////////////////////////////////////

// typedef char int8_t; // already defined ? 
typedef unsigned char uint8_t;

#define VECTOR_SIZE 65536
#define WORD_SIZE 64

//////////////////////////////////////////////

void clear(void* v, int size)
{
  memset(v, 0, size);
}

//////////////////////////////////////////////

DLLEXPORT int cim(int8_t* x, int8_t* w, int* y, uint8_t* count, long* error, long* mean, uint8_t* rpr_table, uint64_t* conf, float* value, uint64_t* dist, uint8_t* dot, int size, int max_rpr, int adc, int R, int C, int NWL, int WL, int NBL, int BL) {

  default_random_engine generator;
  discrete_distribution<uint64_t>** distribution = new discrete_distribution<uint64_t>*[8 * 8 * (max_rpr + 1) * (max_rpr + 1)];
  for (int xb=0; xb<8; xb++) {
    for (int wb=0; wb<8; wb++) {
      for (int wl=0; wl<max_rpr+1; wl++) {
        for (int on=0; on<max_rpr+1; on++) {        
          int xb_addr = xb * 8 * (max_rpr + 1) * (max_rpr + 1);
          int wb_addr =     wb * (max_rpr + 1) * (max_rpr + 1);
          int wl_addr =                     wl * (max_rpr + 1);
          int on_addr =                                     on;
          int addr = xb_addr + wb_addr + wl_addr + on_addr;
          int offset = addr * (adc + 1);
          vector<uint64_t> prob(conf + offset, conf + offset + (adc + 1));
          distribution[addr] = new discrete_distribution<uint64_t>(prob.begin(), prob.end());
        }
      }
    }
  }

  int* pdot = new int[NBL*BL];

  for (int r=0; r<R; r++) {
    for (int wl=0; wl<NWL; wl++) {
      for (int xb=0; xb<8; xb++) {
        for (int wb=0; wb<8; wb++) {

          int wl_ptr = 0;
          int wl_itr = 0;
          while (wl_ptr < WL) {
            int wl_sum = 0;
            clear(pdot, sizeof(int) * NBL*BL);

            int rpr = rpr_table[xb * 8 + wb];
            assert (rpr >= 1);

            while ((wl_ptr < WL) && (wl_sum + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb] <= rpr)) {
              if (x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb]) {
                wl_sum += 1;
                for (int bl=0; bl<NBL*BL; bl+=8) {
                  pdot[bl] += w[(wl * WL * NBL*BL) + (wl_ptr * NBL*BL) + bl + wb];
                }
              }
              wl_ptr += 1;
            }

            for (int bl=0; bl<NBL*BL; bl+=8) {
              int expected = pdot[bl];

              int xb_addr = xb * 8 * (max_rpr + 1) * (max_rpr + 1);
              int wb_addr =     wb * (max_rpr + 1) * (max_rpr + 1);
              int wl_addr =                 wl_sum * (max_rpr + 1);
              int on_addr =                               expected;
              int addr = xb_addr + wb_addr + wl_addr + on_addr;
              int code = (*(distribution[addr]))(generator);
              dist[ (adc + 1) * addr + code ] += 1;

              xb_addr = xb * 8 * (adc + 1);
              wb_addr =     wb * (adc + 1);
              float actual = value[xb_addr + wb_addr + code];
              assert (actual >= 0.);

              if ( ((int) actual) != expected ) { error[xb*8 + wb] += 1; }
              mean[xb*8 + wb] += 1;

              int yaddr = r * C + (bl / 8);
              assert(yaddr < R * C);
              int shift = wb + xb;
              int sign = (wb == 7) ? -1 : 1;
              y[yaddr] += sign * round(actual * pow(2, shift));

              int dot_addr = 0;
              dot_addr += r * NWL * 8 * NBL * BL * size * 3;
              dot_addr +=      wl * 8 * NBL * BL * size * 3;
              dot_addr +=          xb * NBL * BL * size * 3;
              dot_addr +=              (bl + wb) * size * 3;
              dot_addr +=                        wl_itr * 3;
              dot[dot_addr + 0] = wl_sum;
              dot[dot_addr + 1] = expected;
              dot[dot_addr + 2] = code;
            }

            int row_addr = r * NWL * 8 * 8 * size;
            int wl_addr =       wl * 8 * 8 * size;
            int xb_addr =           xb * 8 * size;
            int wb_addr =               wb * size;
            int count_addr = row_addr + wl_addr + xb_addr + wb_addr + wl_itr;
            count[count_addr] = wl_sum;

            assert (wl_itr < size);
            wl_itr += 1;

          } // while (wl_ptr < wl) {
        } // for (int bl=0; bl<BL; bl++) {
      } // for (int xb=0; xb<8; xb++) {
    } // for (int wl=0; wl<WL; wl++) {
  } // for (int r=0; r<R; r++) {

  delete pdot;
  for (int xb=0; xb<8; xb++) {
    for (int wb=0; wb<8; wb++) {
      for (int wl=0; wl<max_rpr+1; wl++) {
        for (int on=0; on<max_rpr+1; on++) {
          int xb_addr = xb * 8 * (max_rpr + 1) * (max_rpr + 1);
          int wb_addr =     wb * (max_rpr + 1) * (max_rpr + 1);
          int wl_addr =                     wl * (max_rpr + 1);
          int on_addr =                                     on;
          int addr = xb_addr + wb_addr + wl_addr + on_addr;
          int offset = addr * (adc + 1);
          delete distribution[addr];
        }
      }
    }
  }
  delete distribution;

  return 1;  
}

































