
#ifndef PIM_H
#define PIM_H

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <assert.h>

typedef unsigned long  uint64_t;
typedef unsigned int   uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char  uint8_t;

typedef long  int64_t;
typedef int   int32_t;
typedef short int16_t;
// typedef char  int8_t;

#define PCM   1

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

//////////////////////////////////////////////

// make sure (bl <= 1024), malloc would be too slow.
// if we just pick a size large enough we will be okay
#define VECTOR_SIZE 288 // number bl per array
#define ARRAY_SIZE 32 // 512 / 16 = 32
#define BLOCK_SIZE 4096 // number of blocks 

//////////////////////////////////////////////

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

#define DYNAMIC   0
#define CENTROIDS 1
#define STATIC    2

/////////////////////////////////////////////////////

long unsigned int factorial(int n);
long unsigned int nChoosek(int n, int k);
float binomial_pmf(int k, int n, float p);
int sat_error(float p, int adc, int rpr);
int eval_adc(float x, int adc, int rpr, int xb, int wb, float* adc_state, float* adc_thresh);
int comps_enabled(int wl, int adc, int rpr, int xb, int wb, float* adc_state, float* adc_thresh);

/////////////////////////////////////////////////////

/*
typedef struct data_t {
  int* x;
  int* w;
  int* y;
} data_t;

typedef struct dim_t {
  int R;
  int B;
  int C;
  int NWL;
  int NBL;
  int WL;
  int BL;
} dim_t;

typedef struct array_t {
  int adc;
  float* adc_state;
  float* adc_thresh;
  float* lut_var;
} array_t;

typedef struct state_t {
  int* r;
  int* next_r;
  int* col;
  int* xb;
  
  int** wl_ptr;
  int** wl_sum;
  int** wl_total;
  
  int*** pdot;
  int*** pdot_sum;
  int*** sat;
} state_t;
*/

typedef struct state_t {
  int* x;
  int* w;
  int* y;

  int R;
  int B;
  int C;
  int NWL;
  int NBL;
  int WL;
  int BL;

  int adc;
  float* adc_state;
  float* adc_thresh;
  
  float* lut_var;
  
  int* r;
  int* next_r;
  int* col;
  int* xb;
  
  int** wl_ptr;
  int** wl_sum;
  int** wl_total;
  
  int*** pdot;
  int*** pdot_sum;
  int*** sat;
} state_t;

/////////////////////////////////////////////////////

class Params {
  public:
  int R;
  int B;
  int C;
  int NWL;
  int NBL;
  int WL;
  int BL;
  
  int D;
  
  int adc;
  int max_rpr;

  float* adc_state;
  float* adc_thresh;

  float* lut_var;
  int* lut_rpr;
  int* lut_bias;
  
  long* metrics;

  int sync;
  int method;
  int skip;

  int ABFT;
  int ABFT_XB;
  int ABFT_ADC;
  
  int XB;
  int COL_PER_ADC;
  int TOTAL_ADC;

  int XB_data;
  int ADC_data;
  int BL_data;
  
  Params(int R, int B, int C, int NWL, int NBL, int WL, int BL, int adc, int max_rpr, float* adc_state, float* adc_thresh, float* lut_var, int* lut_rpr, int* lut_bias, long* metrics, int sync, int method, int skip, int ABFT, int ABFT_XB, int ABFT_ADC);
};

/////////////////////////////////////////////////////

class Array {
  public:
  int block_id;
  int array_id;

  int* x;
  int* w;
  int* y;

  Params* params;

  int xb;
  int col;
  int wl_ptr;
  int wl_sum;
  
  int* pdot;
  int* pdot_adc;

  int* sum_XB;
  int* checksum_XB;
  int* sum_ADC;
  int* checksum_ADC;

  Array(int block_id, int array_id, int* x, int* w, int* y, Params* params);
  
  int pim(int row);
  int pim_skip(int row);
  int pim_base(int row);
  
  int process(int row);
  int collect(int row);
  int correct(int row);
  int update(int row);
  int ABFT(int row);

  int correct_static(int row);

  int clear();
};

/////////////////////////////////////////////////////

class Block {
  public:
  int block_id;
  Params* params;
  Array** arrays;
  int row;

  Block(int block_id, int* x, int* w, int* y, Params* params);
  int pim(int row);
};

/////////////////////////////////////////////////////

class Layer {
  public:
  Params* params;
  int* block_map;
  
  Block** blocks;
  int* row_map;
  int* row_queue;
  
  Layer(int* x, int* w, int* y, Params* params, int* block_map);
  void pim();
  void pim_sync();
};

/////////////////////////////////////////////////////

class LayerSync {
  public:
  Params* params;
  int* block_map;
  
  Block** blocks;
  int* row_map;
  int row_queue;
  
  LayerSync(int* x, int* w, int* y, Params* params, int* block_map);
  void pim();
  void pim_sync();
};

/////////////////////////////////////////////////////

#endif
























