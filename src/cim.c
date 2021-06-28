

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <assert.h>
#include <random>
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

void clear(int* v, int size)
{
  memset(v, 0, sizeof(int) * size);
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

void ecc(int* vec)
{
  // for (int i=0; i<40; i++) assert(vec[i] >= 0);
  
  // localize
  int addr = 0;
  for (int i=0; i<6; i++) { 
    int p = 0;
    for (int j=0; j<32; j++) { 
      p += mask32[i][j] * vec[j];
    }
    p += vec[32 + i];
    addr += pow(2, i) * (p % 2);
  }
  
  // dsum / psum
  int dsum = 0;
  for (int i=0; i<32; i++) dsum += vec[i];
  int psum = 0;
  for (int i=32; i<38; i++) psum += vec[i];
  // ded
  int ded = (dsum + psum + vec[39]) % 2;
  if ((ded == 0) && (addr > 0)) return;
  // sign
  int exp = (dsum + psum) % 4;
  int act = (2*vec[38] + 1*vec[39]) % 4;
  int sign = sign_table[exp][act];
  // correct
  assert (addr < 38);
  int is_data = decode32[addr - 1][0];
  int bit     = decode32[addr - 1][1];
  if (is_data) vec[ 0 + bit] += sign;
  else         vec[32 + bit] += sign;
}

//////////////////////////////////////////////

DLLEXPORT int cim(int8_t* x, int8_t* w, int* y, uint8_t* count, uint8_t* rpr_table, float* var_table, int size, int R, int C, int NWL, int WL, int BL) {
  int* pdot = new int[BL];
  for (int r=0; r<R; r++) {
    for (int wl=0; wl<NWL; wl++) {
      for (int xb=0; xb<8; xb++) {
        for (int wb=0; wb<8; wb++) {

          int wl_ptr = 0;
          int wl_itr = 0;
          while (wl_ptr < WL) {
            int wl_sum = 0;
            clear(pdot, BL);

            int rpr = rpr_table[xb * 8 + wb];
            assert (rpr >= 1);
            while ((wl_ptr < WL) && (wl_sum + x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb] <= rpr)) {
              if (x[(r * NWL * WL * 8) + (wl * WL * 8) + (wl_ptr * 8) + xb]) {
                wl_sum += 1;
                for (int bl=0; bl<BL/8; bl++) {
                  pdot[bl] += w[(wl * WL * BL) + (wl_ptr * BL) + 8*bl + wb];
                }
              }
              wl_ptr += 1;
            }

            for (int bl=0; bl<BL/8; bl++) {
              int key = rand() % 1001;
              int var_addr = (wl_sum * (8 + 1) * 1001) + (pdot[bl] * 1001) + key;
              float var = var_table[var_addr];
              float pdot_var = pdot[bl] + var;

              int pdot_adc;
              if ((pdot_var > 0.20) && (pdot_var < 1.00)) pdot_adc = 1;
              else                                        pdot_adc = min(max((int) round(pdot_var), 0), min(8, rpr));
            
              pdot[bl] = pdot_adc;
            }

            // BL / 8 = 72
            // which means ECC doesnt work here
            // need it to be in the proper shape
            ecc(pdot);

            for (int bl=0; bl<32; bl++) {
              int yaddr = r * C + bl;
              assert(yaddr < R * C);
              int shift = wb + xb;
              int sign = (wb == 7) ? -1 : 1;
              y[yaddr] += sign * (pdot[bl] << shift);
            }

            int row_addr = r * NWL * 8 * 8 * size;
            int wl_addr =       wl * 8 * 8 * size;
            int xb_addr =           xb * 8 * size;
            int wb_addr =               wb * size;
            int count_addr = row_addr + wl_addr + xb_addr + wb_addr + wl_itr;
            count[count_addr] = max(1, wl_sum);

            assert (wl_itr < size);
            wl_itr += 1;
          } // while (wl_ptr < wl) {
        } // for (int bl=0; bl<BL; bl++) {
      } // for (int xb=0; xb<8; xb++) {
    } // for (int wl=0; wl<WL; wl++) {
  } // for (int r=0; r<R; r++) {
  return 1;  
}

































