
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

  this->pdot     = new int[VECTOR_SIZE];
  this->pdot_sum = new int[VECTOR_SIZE];
  this->sat      = new int[VECTOR_SIZE];
}

void Array::pim(int row, int col, int xb, int rpr) {

  memset(this->pdot, 0, sizeof(int) * VECTOR_SIZE);
  this->wl_sum = 0;

  int xaddr = (row * this->params->NWL * this->params->WL * 8) + (this->block_id * this->params->WL * 8) + (this->wl_ptr * 8) + xb;  
  while ((this->wl_ptr < this->params->WL) && (this->wl_sum + this->x[xaddr] <= rpr)) {
    
    if (this->x[xaddr]) {
      this->wl_sum += 1;
      
      for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
        int bl_ptr = adc_ptr + col;
        int waddr = (this->block_id * this->params->WL * this->params->NBL * this->params->BL) + (this->wl_ptr * this->params->NBL * this->params->BL) + (this->array_id * this->params->BL) + bl_ptr;
        this->pdot[bl_ptr] += this->w[waddr];
      }
    }
    
    this->wl_ptr += 1;
    xaddr = (row * this->params->NWL * this->params->WL * 8) + (this->block_id * this->params->WL * 8) + (this->wl_ptr * 8) + xb;  
  }
}
