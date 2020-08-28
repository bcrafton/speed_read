
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

  this->pdot     = new int[BLOCK_SIZE];
  this->pdot_sum = new int[BLOCK_SIZE];
  this->sat      = new int[BLOCK_SIZE];
}


