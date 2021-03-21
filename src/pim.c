
#include "pim.h"
#define DLLEXPORT extern "C"

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

// we need to account for RPR=1 case.
// max output should be 1
// and the threshold for 1 should be very low.
// 4 s.d. of on state.

// should be passing floor thresholds here, not midpoints.
int eval_adc(float x, int adc, int rpr, int xb, int wb, float* adc_state, float* adc_thresh)
{
  assert(adc == 8);
  assert(xb < 8);
  assert(wb < 8);
  // assert(rpr <= 64);

  x = min(x, (float) rpr);

  int offset = xb*adc*(adc+1) + wb*(adc+1);
  for (int i=0; i<=adc; i++) {
    int idx = offset + i;
    if (x < adc_thresh[idx]) {
      return adc_state[idx];
    }
  }
  return adc_state[offset + adc];
}

//////////////////////////////////////////////

int comps_enabled(int wl, int adc, int rpr, int xb, int wb, float* adc_state, float* adc_thresh)
{
  assert(adc == 8);
  assert(xb < 8);
  assert(wb < 8);
  // assert(rpr <= 64);

  int offset = xb*adc*(adc+1) + wb*(adc+1);
  for (int i=1; i<=adc; i++) {
    int idx = offset + i;
    if (wl * 4 <= adc_state[idx]) {
      return i;
    }
  }
  return adc;
}

//////////////////////////////////////////////

DLLEXPORT int pim(int* x, int* w, int* y, float* lut_var, int* lut_rpr, int* lut_bias, long* metrics, int* block_map, float* adc_state, float* adc_thresh, int adc, int max_rpr, int skip, int R, int B, int C, int NWL, int NBL, int WL, int BL, int sync, int method)
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

  Params* params = new Params(R, B, C, NWL, NBL, WL, BL, adc, max_rpr, adc_state, adc_thresh, lut_var, lut_rpr, lut_bias, metrics, sync, method, skip);

  if (sync) {
    LayerSync* layer = new LayerSync(x, w, y, params, block_map);
    layer->pim();
  }
  else {
    Layer* layer = new Layer(x, w, y, params, block_map);
    layer->pim();
  }

  //////////////////////////////

  return metrics[params->adc + METRIC_CYCLE];
}





























