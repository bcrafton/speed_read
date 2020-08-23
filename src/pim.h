
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
#define VECTOR_SIZE 256 // number bl per array
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

#endif
























