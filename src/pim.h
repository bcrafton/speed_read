
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
























