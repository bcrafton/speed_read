
#include "pim.h"

Array::Array(int block_id, int array_id, int* x, int* w, int* y, Params* params) {  
  this->block_id = block_id;
  this->array_id = array_id;

  this->x = x;
  this->w = w;
  this->y = y;
  
  this->params = params;

  this->xb = 0;
  this->col = 0;
  this->wl_ptr = 0;
  this->wl_sum = 0;
  this->pdot = new int[VECTOR_SIZE]();

  this->checksum_WL = new int[VECTOR_SIZE]();
  this->checksum_BL = new int[VECTOR_SIZE]();
}

int Array::clear() {
}

int Array::pim(int row) {
  assert(row < this->params->R);

  if (this->params->skip) {
    return this->pim_skip(row);
  }
  else {
    return this->pim_base(row);
  }
}

int Array::pim_skip(int row) {
  memset(this->pdot, 0, sizeof(int) * VECTOR_SIZE);
  int rpr = this->params->lut_rpr[this->xb * 8 + this->col];

  int xaddr_ROW = (row * this->params->NWL * this->params->WL * 8);
  int xaddr_NWL = (this->block_id * this->params->WL * 8);
  int xaddr_WL  = (this->wl_ptr * 8);
  int xaddr_XB  = xb;
  int xaddr     = xaddr_ROW + xaddr_NWL + xaddr_WL + xaddr_XB;
  assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));
  
  while ((this->wl_ptr < this->params->WL) && ((this->wl_sum + this->x[xaddr]) <= rpr)) {
    assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));
    
    if (this->x[xaddr]) {
      this->wl_sum += 1;
      
      for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
        int bl_ptr = adc_ptr + col;
        int waddr_NWL = (this->block_id * this->params->WL * this->params->NBL * this->params->BL);
        int waddr_WL  = (this->wl_ptr * this->params->NBL * this->params->BL);
        int waddr_NBL = (this->array_id * this->params->BL);
        int waddr_BL  = bl_ptr;
        int waddr     = waddr_NWL + waddr_WL + waddr_NBL + waddr_BL;
        this->pdot[bl_ptr] += this->w[waddr];
      }
    }
    
    this->wl_ptr += 1;
    // careful with placement of this, has to come after wl_ptr update.
    xaddr_ROW = (row * this->params->NWL * this->params->WL * 8);
    xaddr_NWL = (this->block_id * this->params->WL * 8);
    xaddr_WL  = (this->wl_ptr * 8);
    xaddr_XB  = xb;
    xaddr     = xaddr_ROW + xaddr_NWL + xaddr_WL + xaddr_XB;
  }
}

int Array::pim_base(int row) {
}

int Array::process(int row) {
  int rpr = this->params->lut_rpr[this->xb * 8 + this->col];

  int checksum = 0;
  for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
    int bl_ptr = adc_ptr + col;
    int wb = col;

    int key = rand() % 1001;
    int var_addr = this->pdot[bl_ptr] * 1001 + key;
    float var = this->params->lut_var[var_addr];
    float pdot_var = this->pdot[bl_ptr] + var;
    
    int pdot_adc;
    if ((pdot_var > 0.20) && (pdot_var < 1.00)) {
      pdot_adc = 1;
    }
    else {
      pdot_adc = min(max((int) round(pdot_var), 0), min(this->params->adc, rpr));
    }

    if (adc_ptr < this->params->BL_data) {
      checksum += pdot_adc;

      int c = (bl_ptr + this->array_id * this->params->BL_data) / 8; // 8 = COL / ADC ... I think.
      int yaddr = row * this->params->C + c;
      int shift = wb + xb;
      int sign = (wb == 7) ? (-1) : 1;
      this->y[yaddr] += sign * (pdot_adc << shift);
    }
    else {
      assert ((checksum % 2) == (pdot_adc % 2));
    }

  } // for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
}

int Array::collect(int row) {
}

int Array::correct(int row) {
}

int Array::correct_static(int row) {
}

int Array::update(int row) {
  this->wl_sum = 0;
  if (this->wl_ptr == this->params->WL) {
    this->wl_ptr = 0;
    this->xb = (this->xb + 1) % 8;
    if (this->xb == 0) {
      this->col = (this->col + 1) % 8;
      if (this->col == 0) {
        return 1;
      }
    }
  }
  return 0;
}




