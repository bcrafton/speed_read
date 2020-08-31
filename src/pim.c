
#include "pim.h"
#define DLLEXPORT extern "C"

//////////////////////////////////////////////

void clear_vector(int* v)
{
  memset(v, 0, sizeof(int) * VECTOR_SIZE);
}

void clear_array(int* a)
{
  memset(a, 0, sizeof(int) * ARRAY_SIZE);
}

void clear_block(int* a)
{
  memset(a, 0, sizeof(int) * BLOCK_SIZE);
}

//////////////////////////////////////////////

void free3D(int*** array)
{
  for (int i=0; i<BLOCK_SIZE; i++) {
    for (int j=0; j<ARRAY_SIZE; j++) {
      free(array[i][j]);
    }
    free(array[i]);
  }
  free(array);
}

//////////////////////////////////////////////

int* array1D()
{
  int* array = (int*) malloc(sizeof(int) * BLOCK_SIZE);
  clear_block(array);
  return array;
}

int** array2D()
{
  int** array = (int**) malloc(sizeof(int*) * BLOCK_SIZE);
  for (int i=0; i<BLOCK_SIZE; i++) {
    array[i] = (int*) malloc(sizeof(int) * ARRAY_SIZE);
    clear_array(array[i]);
  }
  return array;
}

int*** array3D()
{
  int*** array = (int***) malloc(sizeof(int**) * BLOCK_SIZE);
  for (int i=0; i<BLOCK_SIZE; i++) {
    array[i] = (int**) malloc(sizeof(int*) * ARRAY_SIZE);
    for (int j=0; j<ARRAY_SIZE; j++) {
      array[i][j] = (int*) malloc(sizeof(int) * VECTOR_SIZE);
      clear_vector(array[i][j]);
    }
  }
  return array;
}

//////////////////////////////////////////////

long unsigned int factorial(int n)
{
  long unsigned int fact = 1;
  for (int i=1; i<(n+1); i++) {
    fact = fact * i;
  }
  return fact;
}

long unsigned int nChoosek(int n, int k)
{
  long unsigned int t = factorial(n);
  long unsigned int b = factorial(k) * factorial(n - k);
  long unsigned int nCk = t / b;
  assert (nCk > 0);
  return nCk;
}

float binomial_pmf(int k, int n, float p)
{
  long unsigned int nCk = nChoosek(n, k);
  float success = pow(p, k);
  float fail = pow(1 - p, n - k);
  float pmf = nCk * success * fail;
  assert (pmf >= 0.);
  return pmf;
}

int sat_error(float p, int adc, int rpr)
{
  if (rpr <= adc) {
    return 0;
  }
  float e = 0.;
  float bin_sum = 0.;
  for (int s=adc; s<(rpr+1); s++) {
    float bin = binomial_pmf(s, rpr, p);
    bin_sum += bin;
    e += bin * (adc - s);
  }
  assert (bin_sum >= 0.);
  if (bin_sum > 0.) {
    e /= bin_sum;
  }
  assert (e <= 0);
  return e;
}

//////////////////////////////////////////////

/*
int eval_adc(float x, int adc, int rpr, float* adc_thresh)
{
  int offset = rpr * adc;

  int minarg = offset;
  float mindiff = abs(x - adc_thresh[minarg]);
  
  for (int i=1; i<adc; i++) {
    int idx = offset + i;  
    float diff = abs(x - adc_thresh[idx]);
    minarg = (diff < mindiff) ? idx : minarg;
    mindiff = (diff < mindiff) ? diff : mindiff;
    
    // printf("%d %d: %f %f\n", i, rpr, x, adc_thresh[idx]);
  }
  
  return adc_thresh[minarg];
}
*/

//////////////////////////////////////////////

// we need to account for RPR=1 case.
// max output should be 1
// and the threshold for 1 should be very low.
// 4 s.d. of on state.

// should be passing floor thresholds here, not midpoints.
int eval_adc(float x, int adc, int rpr, float* adc_state, float* adc_thresh)
{
  assert(adc == 8);

  x = min(x, (float) rpr);

  int offset = rpr * (adc + 1);
  for (int i=0; i<=adc; i++) {
    int idx = offset + i;
    if (x < adc_thresh[idx]) {
      return adc_state[idx];
    }
  }
  return adc_state[offset + adc];
}

//////////////////////////////////////////////

int comps_enabled(int wl, int adc, int rpr, float* adc_state, float* adc_thresh)
{
  assert(adc == 8);

  int offset = rpr * (adc + 1);

  for (int i=1; i<=adc; i++) {
    int idx = offset + i;
    if (wl * 4 <= adc_state[idx]) {
      return i;
    }
  }
  return adc;
}

//////////////////////////////////////////////

void pim_kernel(state_t* s, int rpr, int block, int wl, int bl) {

  clear_vector(s->pdot[block][bl]);
  s->wl_sum[block][bl] = 0;

  int xaddr = (s->r[block] * s->NWL * s->WL * 8) + (wl * s->WL * 8) + (s->wl_ptr[block][bl] * 8) + s->xb[block];  
  
  while ((s->wl_ptr[block][bl] < s->WL) && (s->wl_sum[block][bl] + s->x[xaddr] <= rpr)) {
    
    if (s->x[xaddr]) {
      s->wl_sum[block][bl] += 1;
      
      for (int adc_ptr=0; adc_ptr<s->BL; adc_ptr+=8) {
        int bl_ptr = adc_ptr + s->col[block];
        int waddr = (wl * s->WL * s->NBL * s->BL) + (s->wl_ptr[block][bl] * s->NBL * s->BL) + (bl * s->BL) + bl_ptr;
        s->pdot[block][bl][bl_ptr] += s->w[waddr];
      }
    }
    s->wl_ptr[block][bl] += 1;
    xaddr = (s->r[block] * s->NWL * s->WL * 8) + (wl * s->WL * 8) + (s->wl_ptr[block][bl] * 8) + s->xb[block];
  }

}

//////////////////////////////////////////////

void process(state_t* s, int rpr, int block, int bl) 
{
  for (int adc_ptr=0; adc_ptr<s->BL; adc_ptr+=8) {
    int bl_ptr = adc_ptr + s->col[block];
    int c = (bl_ptr + bl * s->BL) / 8;
    int wb = s->col[block];

    int key = rand() % 1000;
    int var_addr = s->pdot[block][bl][bl_ptr] * 1000 + key;
    float var = s->lut_var[var_addr];

    float pdot_var = s->pdot[block][bl][bl_ptr] + var;
    // int pdot_adc = eval_adc(pdot_var, s->adc, rpr, s->adc_state, s->adc_thresh);
    int pdot_adc = min(max((int) round(pdot_var), 0), min(s->adc, rpr));

    int yaddr = s->r[block] * s->C + c;
    int shift = wb + s->xb[block];
    s->y[yaddr] += pdot_adc << shift;
    
    if (wb == 0) { 
      // s->y[yaddr] -= 4 * ((s->wl_sum[block][bl] * 128) << s->xb[block]); 
      s->y[yaddr] -= ((s->wl_sum[block][bl] * 128) << s->xb[block]); 
    }
   
    if (s->wl_sum[block][bl] >= s->adc) {
      s->sat[block][bl][bl_ptr] += (s->pdot[block][bl][bl_ptr] == s->adc);
      s->pdot_sum[block][bl][bl_ptr] += s->pdot[block][bl][bl_ptr];
    }
    
  }
}

//////////////////////////////////////////////

void correct(state_t* s, int rpr, int block, int bl) 
{
  if (s->wl_ptr[block][bl] == s->WL) {
    for (int adc_ptr=0; adc_ptr<s->BL; adc_ptr+=8) {
      int bl_ptr = adc_ptr + s->col[block];
      int c = (bl_ptr + bl * s->BL) / 8;
      int wb = s->col[block];

      if (s->wl_total[block][bl]) {
        float p = ((float) s->pdot_sum[block][bl][bl_ptr]) / ((float) s->wl_total[block][bl]);
        p = min(max(p, 0.), 1.);
        int e = sat_error(p, s->adc, rpr);
        
        int yaddr = s->r[block] * s->C + c;
        s->y[yaddr] -= ((s->sat[block][bl][bl_ptr] * e) << (wb + s->xb[block]));
        
        s->sat[block][bl][bl_ptr] = 0;
        s->pdot_sum[block][bl][bl_ptr] = 0;
      }
    }
  }
}

//////////////////////////////////////////////

void collect(state_t* s, long* metrics, int rpr, int block, int bl) 
{
  for (int adc_ptr=0; adc_ptr<s->BL; adc_ptr+=8) {
    int bl_ptr = adc_ptr + s->col[block];
    int c = (bl_ptr + bl * s->BL) / 8;
    int wb = s->col[block];
    
    metrics[METRIC_RON] += s->pdot[block][bl][bl_ptr];
    metrics[METRIC_ROFF] += s->wl_sum[block][bl] - s->pdot[block][bl][bl_ptr];
  }

  if (s->wl_sum[block][bl] > 0) {
    // int comps = comps_enabled(s->wl_sum[block][bl], s->adc, rpr, s->adc_state, s->adc_thresh) - 1;
    int comps = min(s->wl_sum[block][bl] - 1, s->adc - 1);
    assert((comps >= 0) && (comps < s->adc));
    assert ((s->BL % 8) == 0);
    metrics[comps] += s->BL / 8;
  }

  metrics[METRIC_WL] += s->wl_sum[block][bl];
}

//////////////////////////////////////////////

DLLEXPORT int pim(int* x, int* w, int* y, float* lut_var, int* lut_rpr, long* metrics, int* block_map, float* adc_state, float* adc_thresh, int adc, int skip, int R, int B, int C, int NWL, int NBL, int WL, int BL)
{
  // x = nrow, nwl, wl, xb
  // f = nwl, wl, nbl, bl
  // y = nrow, ncol
  
  // our arrays are sized for 128. need to increase.
  // assert ((D >= 1) && (NWL >= 1) && (NBL >= 1) && (BL >= 1));
  // assert (NWL >= 1) && (NBL >= 1) && (BL >= 1));
  assert (NWL >= 1);
  assert (NBL >= 1);
  assert (BL >= 1);
  assert (NBL <= ARRAY_SIZE);
  assert (BL <= VECTOR_SIZE);
  assert (B <= BLOCK_SIZE);
  
  //////////////////////////////

  Params* params = new Params(R, B, C, NWL, NBL, WL, BL, adc, adc_state, adc_thresh, lut_var, lut_rpr, metrics);

  Layer* layer = new Layer(x, w, y, params, block_map);
  layer->pim_sync();

  //////////////////////////////

  return metrics[METRIC_CYCLE];
}





























