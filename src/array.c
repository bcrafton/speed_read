
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
}

int Array::process(int row, int col, int xb, int rpr) {

  int checksum = 0;
  for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
    int bl_ptr = adc_ptr + col;
    int wb = col;

    int key = rand() % 1001;
    int var_addr = this->pdot[bl_ptr] * 1001 + key;
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

    if (adc_ptr < this->params->BL_data) {
      checksum += pdot_adc;

      int c = (bl_ptr + this->array_id * this->params->BL_data) / 8; // 8 = COL / ADC ... I think.
      int yaddr = row * this->params->C + c;
      int shift = wb + xb;
      int sign = (wb == 7) ? -1 : 1;
      this->y[yaddr] += sign * (pdot_adc << shift);
    }
    else {
      // printf("%d %d\n", checksum % 2, pdot_adc % 2);
      assert ((checksum % 2) == (pdot_adc % 2));
    }

  } // for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
}

int Array::collect(int row, int col, int xb, int rpr) {
}

int Array::correct(int row, int col, int xb, int rpr) {
}

int Array::correct_static(int row, int col, int xb, int rpr) {
}



