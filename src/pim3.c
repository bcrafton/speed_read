
#include "pim.h"

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
    int pdot_adc = eval_adc(pdot_var, s->adc, rpr, s->adc_state, s->adc_thresh);

    int yaddr = s->r[block] * s->C + c;
    int shift = wb + s->xb[block];
    s->y[yaddr] += pdot_adc << shift;
    
    if (wb == 0) { 
      s->y[yaddr] -= 4 * ((s->wl_sum[block][bl] * 128) << s->xb[block]); 
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

  if (s->wl_sum[block][bl] == 0) {
  }
  else {
    int comps = comps_enabled(s->wl_sum[block][bl], s->adc, rpr, s->adc_state, s->adc_thresh) - 1;
    assert((comps >= 0) && (comps < s->adc));
    assert ((s->BL % 8) == 0);
    metrics[comps] += s->BL / 8;
  }

  metrics[METRIC_WL] += s->wl_sum[block][bl];
}

//////////////////////////////////////////////

int sync(state_t* s, int rpr, int block, int wl, int bl) {

  int done = 0;
  int* block_done = array1D();

  if (s->wl_ptr[block][bl] == s->WL) {
    s->wl_ptr[block][bl] = 0;
    s->wl_total[block][bl] = 0;

    if (bl == (s->NBL - 1)) {
      if (s->col[block] == (8 - 1)) {
        s->col[block] = 0;
    
        if (s->xb[block] == (8 - 1)) {
          s->xb[block] = 0;
          
          if (s->next_r[wl] < s->R) {
            s->r[block] = s->next_r[wl];
            s->next_r[wl]++;
          }
          else {
            block_done[block] = 1;
            
            int block_sync = 1;
            for (int i=0; i<s->B; i++) {
              block_sync = block_sync & block_done[i];
            }

            done = block_sync;
          }
        }
        else {
          s->xb[block] += 1;
        }
      }
      else {
        s->col[block] += 1;
      }
    }
  }
  else {
    assert (s->wl_ptr[block][bl] < s->WL);
  }
  return done;
}

//////////////////////////////////////////////

int pim(int* x, int* w, int* y, float* lut_var, int* lut_rpr, long* metrics, int* block_map, float* adc_state, float* adc_thresh, int adc, int skip, int R, int B, int C, int NWL, int NBL, int WL, int BL)
{
  // x = nrow, nwl, wl, xb
  // f = nwl, wl, nbl, bl
  // y = nrow, ncol
  
  // our arrays are sized for 128. need to increase.
  // assert ((D >= 1) && (NWL >= 1) && (NBL >= 1) && (BL >= 1));
  assert ((NWL >= 1) && (NBL >= 1) && (BL >= 1));
  assert (NBL <= ARRAY_SIZE);
  assert (BL <= VECTOR_SIZE);
  assert (B <= BLOCK_SIZE);
  
  int done = 0;
    
  int* r = array1D(); 
  int* next_r = array1D(); 
    
  int* block_done = array1D();
  int* col = array1D();
  int* xb = array1D();

  int** wl_ptr = array2D();
  int** wl_sum = array2D();
  int** wl_total = array2D();
  
  int*** pdot = array3D();
  int*** pdot_sum = array3D();
  int*** sat = array3D();
  
  //////////////////////////////
  
  // break this up somehow.
  state_t state = {x, w, y, R, B, C, NWL, NBL, WL, BL, adc, adc_state, adc_thresh, lut_var, r, next_r, col, xb, wl_ptr, wl_sum, wl_total, pdot, pdot_sum, sat};
  
  //////////////////////////////

  for (int block=0; block<B; block++) {
    int wl = block_map[block];
    assert (wl < NWL);
    r[block] = next_r[wl];
    next_r[wl]++;
  }  
  // next_r will have to be 2D.
  // all {row, wl}

  while (!done) {

    for (int i=0; i<B; i++) {
      for (int j=0; j<NBL; j++) {
        assert(wl_ptr[i][0] == wl_ptr[i][j]);
      }
    }

    metrics[METRIC_CYCLE] += 1;
    // if there are more duplicates than rows, then I believe we hit this assert.
    // assert (metrics[METRIC_CYCLE] < 500000);

    for (int block=0; block<B; block++) {
      
      int wl = block_map[block];
      assert (wl < NWL);

      // here is our issue.
      if (block_done[block]) {
        metrics[METRIC_STALL] += NBL;
        continue;
      }
      else { 
        metrics[METRIC_BLOCK_CYCLE + wl] += 1;
      }
      
      for (int bl=0; bl<NBL; bl++) {
        
        int x_addr = xb[block] * 8;
        int w_addr = col[block];
        int rpr_addr = x_addr + w_addr;

        if (!((rpr_addr >= 0) && (rpr_addr < 64))) {
          printf("xb: %d bl: %d BL: %d C: %d: rpr_addr: %d\n", xb[block], bl, BL, C, rpr_addr);
          assert ((rpr_addr >= 0) && (rpr_addr < 64));
        }
        int rpr = lut_rpr[rpr_addr];
        assert (rpr >= 1);
                
        /////////////////////////////////////
        
        pim_kernel(&state, rpr, block, wl, bl);
        process(&state, rpr, block, bl);
        collect(&state, metrics, rpr, block, bl);
        done = sync(&state, rpr, block, wl, bl);
        
        /////////////////////////////////////

      } // for (int bl=0; bl<NBL; bl++) {
    } // for (int b=0; b<B; b++) {
  } // while (!done) {

  free3D(pdot);
  free3D(pdot_sum);
  free3D(sat);

  return metrics[METRIC_CYCLE];  
}





























