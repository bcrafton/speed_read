

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

int decode32[38][2] = {
{ 0,  0},
{ 0,  1},
{ 1,  0},
{ 0,  2},
{ 1,  1},
{ 1,  2},
{ 1,  3},
{ 0,  3},

{ 1,  4},
{ 1,  5},
{ 1,  6},
{ 1,  7},
{ 1,  8},
{ 1,  9},
{ 1, 10},
{ 0,  4},

{ 1, 11},
{ 1, 12},
{ 1, 13},
{ 1, 14},
{ 1, 15},
{ 1, 16},
{ 1, 17},
{ 1, 18},

{ 1, 19},
{ 1, 20},
{ 1, 21},
{ 1, 22},
{ 1, 23},
{ 1, 24},
{ 1, 25},
{ 0,  5},

{ 1, 26},
{ 1, 27},
{ 1, 28},
{ 1, 29},
{ 1, 30},
{ 1, 31}
};

//////////////////////////////////////////////

int mask32[6][32] = {
{1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0},
{1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1},
{0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1},
{0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}
};

//////////////////////////////////////////////

int sign_table[4][4] = {
{ 0,  1,  0, -1},
{-1,  0,  1,  0},
{ 0, -1,  0,  1},
{ 1,  0, -1,  0}
};

//////////////////////////////////////////////

// TODO: we need a test bench for this.
int ecc(int* data, int* parity)
{
  // localize
  int addr = 0;
  for (int i=0; i<6; i++) { 
    int p = 0;
    for (int j=0; j<32; j++) { 
      p += mask32[i][j] * data[j];
    }
    p += parity[i];
    addr += pow(2, i) * (p % 2);
  }
  if (addr > 38) return 0;
  if (addr == 0) return 0;

  // dsum / psum
  int dsum = 0;
  for (int i=0; i<32; i++) dsum += data[i];
  int psum = 0;
  for (int i=0; i<6; i++) psum += parity[i];
  // ded
  int ded = (dsum + psum + parity[7]) % 2;
  if ((ded == 0) && (addr > 0)) return 0;
  // sign
  int exp = (dsum + psum) % 4;
  int act = (2*parity[6] + 1*parity[7]) % 4;
  int sign = sign_table[exp][act];
  // correct
  int is_data = decode32[addr - 1][0];
  int bit     = decode32[addr - 1][1];
  if (is_data) data[bit]   += sign;
  else         parity[bit] += sign;
  return (addr > 0);
}

//////////////////////////////////////////////

DLLEXPORT int cim(int8_t* x, int8_t* w, int8_t* p, int* y, uint8_t* count, uint32_t* error, uint8_t* rpr_table, uint8_t* step_table, uint32_t* conf, float* value, int size, int max_rpr, int adc, int R, int C, int NWL, int WL, int NBL, int BL, int BL_P) {

  default_random_engine generator;
  discrete_distribution<uint32_t>* distribution = new discrete_distribution<uint32_t>[8 * 8 * (max_rpr + 1) * (max_rpr + 1)];
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
          vector<uint32_t> prob(conf + offset, conf + offset + (adc + 1));
          distribution[addr] = discrete_distribution<uint32_t>(prob.begin(), prob.end());
        }
      }
    }
  }

  int* pdot     = new int[NBL*BL];
  int* pdot_ecc = new int[NBL*BL_P];
  
  float* adc_code     = new float[NBL*BL];
  float* adc_code_ecc = new float[NBL*BL_P];

  int correct = 0;

  for (int r=0; r<R; r++) {
    for (int wl=0; wl<NWL; wl++) {
      for (int xb=0; xb<8; xb++) {
        for (int wb=0; wb<8; wb++) {

          int wl_ptr = 0;
          int wl_itr = 0;
          while (wl_ptr < WL) {
            int wl_sum = 0;
            clear(pdot,         sizeof(int)   * NBL*BL);
            clear(pdot_ecc,     sizeof(int)   * NBL*BL_P);
            clear(adc_code,     sizeof(float) * NBL*BL);
            clear(adc_code_ecc, sizeof(float) * NBL*BL_P);

            int rpr = rpr_table[xb * 8 + wb];
            int step = step_table[xb * 8 + wb];
            assert (rpr >= 1);
            while ((wl_ptr < WL) && (wl_sum + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb] <= rpr)) {
              if (x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb]) {
                wl_sum += 1;
                for (int bl=0; bl<NBL*BL/8; bl++) {
                  pdot[bl] += w[(wl * WL * NBL*BL) + (wl_ptr * NBL*BL) + 8*bl + wb];
                }
                for (int bl=0; bl<NBL*BL_P/8; bl++) {
                  pdot_ecc[bl] += p[(wl * WL * (NBL*BL_P)) + (wl_ptr * (NBL*BL_P)) + 8*bl + wb];
                }
              }
              wl_ptr += 1;
            }

            for (int nbl=0; nbl<NBL; nbl++) {
              for (int bl=0; bl<BL/8; bl++) {
                int expected = pdot[BL/8 * nbl + bl];

                int xb_addr = xb * 8 * (max_rpr + 1) * (max_rpr + 1);
                int wb_addr =     wb * (max_rpr + 1) * (max_rpr + 1);
                int wl_addr =                 wl_sum * (max_rpr + 1);
                int on_addr =                               expected;
                int addr = xb_addr + wb_addr + wl_addr + on_addr;
                int code = distribution[addr](generator);
                assert (code <= (adc / pow(2, step)));

                xb_addr = xb * 8 * (adc + 1);
                wb_addr =     wb * (adc + 1);
                float actual = value[xb_addr + wb_addr + code];
                assert (actual >= 0.);
                adc_code[BL/8 * nbl + bl] = actual;
              }
            }
            
            for (int bl=0; bl<NBL*BL/8; bl++) {
              int yaddr = r * C + bl;
              assert(yaddr < R * C);
              int shift = wb + xb;
              int sign = (wb == 7) ? -1 : 1;
              y[yaddr] += sign * adc_code[bl] * pow(2, shift);
            }

            int row_addr = r * NWL * 8 * 8 * size;
            int wl_addr =       wl * 8 * 8 * size;
            int xb_addr =           xb * 8 * size;
            int wb_addr =               wb * size;
            int count_addr = row_addr + wl_addr + xb_addr + wb_addr + wl_itr;
            // count[count_addr] = max(1, wl_sum);
            count[count_addr] = wl_sum;

            assert (wl_itr < size);
            wl_itr += 1;
          } // while (wl_ptr < wl) {
        } // for (int bl=0; bl<BL; bl++) {
      } // for (int xb=0; xb<8; xb++) {
    } // for (int wl=0; wl<WL; wl++) {
  } // for (int r=0; r<R; r++) {
  // printf("%d\n", correct);
  return 1;  
}

































