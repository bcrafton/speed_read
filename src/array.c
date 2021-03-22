
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

int Array::clear() {
  this->wl_ptr = 0;
  this->wl_total = 0;
}

int Array::pim(int row, int col, int xb, int rpr) {
  assert(row < this->params->R);

  if (this->params->skip) {
    return this->pim_skip(row, col, xb, rpr);
  }
  else {
    return this->pim_base(row, col, xb, rpr);
  }
}

int Array::pim_skip(int row, int col, int xb, int rpr) {

  memset(this->pdot, 0, sizeof(int) * VECTOR_SIZE);
  this->wl_sum = 0;

  int xaddr = (row * this->params->NWL * this->params->WL * 8) + (this->block_id * this->params->WL * 8) + (this->wl_ptr * 8) + xb;
  assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));
  
  while ((this->wl_ptr < this->params->WL) && ((this->wl_sum + this->x[xaddr]) <= rpr)) {
    assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));
    
    if (this->x[xaddr]) {
      this->wl_sum += 1;
      
      for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
        int bl_ptr = adc_ptr + col;
        int waddr = (this->block_id * this->params->WL * this->params->NBL * this->params->BL) + (this->wl_ptr * this->params->NBL * this->params->BL) + (this->array_id * this->params->BL) + bl_ptr;
        this->pdot[bl_ptr] += this->w[waddr];
      }
    }
    
    this->wl_ptr += 1;
    // careful with placement of this, has to come after wl_ptr update.
    xaddr = (row * this->params->NWL * this->params->WL * 8) + (this->block_id * this->params->WL * 8) + (this->wl_ptr * 8) + xb;
  }

  if (this->wl_sum >= this->params->adc) {
    this->wl_total += this->wl_sum;
  }

  if (this->wl_ptr == this->params->WL) {
    return 1;
  }
  return 0;
}

int Array::pim_base(int row, int col, int xb, int rpr) {

  memset(this->pdot, 0, sizeof(int) * VECTOR_SIZE);
  this->wl_sum = 0;

  int xaddr = (row * this->params->NWL * this->params->WL * 8) + (this->block_id * this->params->WL * 8) + (this->wl_ptr * 8) + xb;
  assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));
  
  // while ((this->wl_ptr < this->params->WL) && ((this->wl_sum + this->x[xaddr]) <= rpr)) {
  int start = this->wl_ptr;
  while ((this->wl_ptr < this->params->WL) && (this->wl_ptr < (start + this->params->adc))) {
    assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));
    
    if (this->x[xaddr]) {
      this->wl_sum += 1;
      
      for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
        int bl_ptr = adc_ptr + col;
        int waddr = (this->block_id * this->params->WL * this->params->NBL * this->params->BL) + (this->wl_ptr * this->params->NBL * this->params->BL) + (this->array_id * this->params->BL) + bl_ptr;
        this->pdot[bl_ptr] += this->w[waddr];
      }
    }
    
    this->wl_ptr += 1;
    // careful with placement of this, has to come after wl_ptr update.
    xaddr = (row * this->params->NWL * this->params->WL * 8) + (this->block_id * this->params->WL * 8) + (this->wl_ptr * 8) + xb;
  }

  if (this->wl_sum >= this->params->adc) {
    this->wl_total += this->wl_sum;
  }

  if (this->wl_ptr == this->params->WL) {
    return 1;
  }
  return 0;
}

int Array::process(int row, int col, int xb, int rpr) {

  for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
    int bl_ptr = adc_ptr + col;
    int c = (bl_ptr + this->array_id * this->params->BL) / 8;
    int wb = col;

    int key = rand() % 1001;
    int var_addr = (this->wl_sum * (params->max_rpr + 1) * 1001) + (this->pdot[bl_ptr] * 1001) + key;
    int var_size = (params->max_rpr + 1) * (params->max_rpr + 1) * 1001;
    assert (var_addr < var_size);
    float var = this->params->lut_var[var_addr];

    float pdot_var = this->pdot[bl_ptr] + var;
    int pdot_adc;

    if (params->method == CENTROIDS) {
      pdot_adc = eval_adc(pdot_var, this->params->adc, rpr, xb, wb, this->params->adc_state, this->params->adc_thresh);
    }
    else {
      if ((pdot_var > 0.20) && (pdot_var < 1.00)) {
        pdot_adc = 1;
      }
      else {
        pdot_adc = min(max((int) round(pdot_var), 0), min(this->params->adc, rpr));
      }
    }

    int yaddr = row * this->params->C + c;
    int shift = wb + xb;
    this->y[yaddr] += pdot_adc << shift;

    if (wb == 0) {
      if (params->method == CENTROIDS) this->y[yaddr] -= 4 * ((this->wl_sum * 128) << xb);
      else                             this->y[yaddr] -= ((this->wl_sum * 128) << xb);
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
    int wb = col;

    if (params->skip == 0) {
      if (this->wl_sum > 0) {
        comps = this->params->adc - 1;
      }
      else {
        comps = 0;
      }
    }
    else if (params->method == CENTROIDS) comps = comps_enabled(this->wl_sum, this->params->adc, rpr, xb, wb, this->params->adc_state, this->params->adc_thresh) - 1;
    else                                  comps = min(this->wl_sum - 1, this->params->adc - 1);
    assert((comps >= 0) && (comps < this->params->adc));
    assert ((this->params->BL % 8) == 0);

    int adc_offset    = METRIC_BLOCK_CYCLE + params->NWL;
    int xb_address    = xb * 8 * params->NWL    * (params->adc + 1);
    int wb_address    =     wb * params->NWL    * (params->adc + 1);
    int block_address =          this->block_id * (params->adc + 1);
    int comp_address  =                                       comps;
    int address = adc_offset + xb_address + wb_address + block_address + comp_address;
    this->params->metrics[address] += this->params->BL / 8;
  }

  this->params->metrics[METRIC_WL] += this->wl_sum;
}

int Array::correct(int row, int col, int xb, int rpr) {
  assert (this->wl_ptr == this->params->WL);

  if (this->wl_total) {
    for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
      int bl_ptr = adc_ptr + col;
      int c = (bl_ptr + this->array_id * this->params->BL) / 8;
      int wb = col;

      float p = ((float) this->pdot_sum[bl_ptr]) / ((float) this->wl_total);
      p = min(max(p, 0.), 1.);
      int e = sat_error(p, this->params->adc, rpr);

      int yaddr = row * this->params->C + c;
      int bias = this->sat[bl_ptr] * e;
      assert (bias <= 0.);
      this->y[yaddr] -= (bias << (wb + xb));

      this->sat[bl_ptr] = 0;
      this->pdot_sum[bl_ptr] = 0;
    }
  }

}

int Array::correct_static(int row, int col, int xb, int rpr) {
  assert (this->wl_ptr == this->params->WL);

  for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
    int bl_ptr = adc_ptr + col;
    int c = (bl_ptr + this->array_id * this->params->BL) / 8;
    int wb = col;

    int yaddr = row * this->params->C + c;
    // int bias = (this->sat[bl_ptr] * this->params->lut_bias[rpr]) / 256;
    float bias_float = (this->sat[bl_ptr] * this->params->lut_bias[8 * xb + wb]) / 256.;
    int bias = (int) round(bias_float);
    assert (bias >= 0.);
    this->y[yaddr] += (bias << (wb + xb));

    // if (this->sat[bl_ptr]) {printf("%d %d %d %d\n", xb, wb, rpr, this->params->lut_bias[8 * xb + wb]);}
    // printf("%d %d\n", this->sat[bl_ptr], this->params->lut_bias[rpr]);

    this->sat[bl_ptr] = 0;
    this->pdot_sum[bl_ptr] = 0;
  }
}



