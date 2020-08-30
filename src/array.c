
#include "pim.h"

Array::Array(int block_id, int array_id, int* x, int* w, int* y, Params* params) {  
  this->block_id = block_id;
  this->array_id = array_id;

  this->x = x;
  this->w = w;
  this->y = y;
  
  this->params = params;

  this->wl_ptr = 0;
  this->wl_sum = 0;
  this->wl_total = 0;

  this->pdot     = new int[VECTOR_SIZE]();
  this->pdot_sum = new int[VECTOR_SIZE]();
  this->sat      = new int[VECTOR_SIZE]();
}

int Array::pim(int row, int col, int xb, int rpr) {

  memset(this->pdot, 0, sizeof(int) * VECTOR_SIZE);
  this->wl_sum = 0;

  int xaddr = (row * this->params->NWL * this->params->WL * 8) + (this->block_id * this->params->WL * 8) + (this->wl_ptr * 8) + xb;
  assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));
  
  while ((this->wl_ptr < this->params->WL) && ((this->wl_sum + this->x[xaddr]) <= rpr)) {
    
    if (this->x[xaddr]) {
      this->wl_sum += 1;
      
      for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
        int bl_ptr = adc_ptr + col;
        int waddr = (this->block_id * this->params->WL * this->params->NBL * this->params->BL) + (this->wl_ptr * this->params->NBL * this->params->BL) + (this->array_id * this->params->BL) + bl_ptr;
        this->pdot[bl_ptr] += this->w[waddr];
      }
    }
        
    xaddr = (row * this->params->NWL * this->params->WL * 8) + (this->block_id * this->params->WL * 8) + (this->wl_ptr * 8) + xb;
    // printf("%d %d %d %d %d\n", row, this->block_id, this->wl_ptr, xb, this->x[xaddr]);
    assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));

    this->wl_ptr += 1;
  }
    
  if (this->wl_ptr == this->params->WL) {
    this->wl_ptr = 0;
    return 1;
  }
  return 0;
}

#define CENTROIDS 0

int Array::process(int row, int col, int xb, int rpr) {

  for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
    int bl_ptr = adc_ptr + col;
    int c = (bl_ptr + this->array_id * this->params->BL) / 8;
    int wb = col;

    int key = rand() % 1000;
    int var_addr = this->pdot[bl_ptr] * 1000 + key;
    float var = this->params->lut_var[var_addr];

    float pdot_var = this->pdot[bl_ptr] + var;
    int pdot_adc;

    if (CENTROIDS) pdot_adc = eval_adc(pdot_var, this->params->adc, rpr, this->params->adc_state, this->params->adc_thresh);
    else           pdot_adc = min(max((int) round(pdot_var), 0), min(this->params->adc, rpr));

    int yaddr = row * this->params->C + c;
    int shift = wb + xb;
    this->y[yaddr] += pdot_adc << shift;

    if (wb == 0) {
      if (CENTROIDS) this->y[yaddr] -= 4 * ((this->wl_sum * 128) << xb);
      else           this->y[yaddr] -= ((this->wl_sum * 128) << xb);
    }

    if (this->wl_sum >= this->params->adc) {
      this->sat[bl_ptr] += (this->pdot[bl_ptr] == this->params->adc);
      this->pdot_sum[bl_ptr] += this->pdot[bl_ptr];
    }

  }
}

int Array::collect(int row, int col, int xb, int rpr) {

  for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
    int bl_ptr = adc_ptr + col;
    int c = (bl_ptr + this->array_id * this->params->BL) / 8;
    int wb = col;

    this->params->metrics[METRIC_RON] += this->pdot[bl_ptr];
    this->params->metrics[METRIC_ROFF] += this->wl_sum - this->pdot[bl_ptr];
  }

  if (this->wl_sum > 0) {
    int comps;
    if (CENTROIDS) comps = comps_enabled(this->wl_sum, this->params->adc, rpr, this->params->adc_state, this->params->adc_thresh) - 1;
    else           comps = min(this->wl_sum - 1, this->params->adc - 1);
    assert((comps >= 0) && (comps < this->params->adc));
    assert ((this->params->BL % 8) == 0);
    this->params->metrics[comps] += this->params->BL / 8;
  }

  this->params->metrics[METRIC_WL] += this->wl_sum;
}

int Array::correct(int row, int col, int xb, int rpr) {

  if (this->wl_ptr == this->params->WL) {
    for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
      int bl_ptr = adc_ptr + col;
      int c = (bl_ptr + this->array_id * this->params->BL) / 8;
      int wb = col;

      if (this->wl_total) {
        float p = ((float) this->pdot_sum[bl_ptr]) / ((float) this->wl_total);
        p = min(max(p, 0.), 1.);
        int e = sat_error(p, this->params->adc, rpr);

        int yaddr = row * this->params->C + c;
        this->y[yaddr] -= ((this->sat[bl_ptr] * e) << (wb + xb));

        this->sat[bl_ptr] = 0;
        this->pdot_sum[bl_ptr] = 0;
      }
    }
  }

}






