
#include "pim.h"

Block::Block(int block_id, int size, int* x, int* w, int* y, Params* params) {
  this->block_id = block_id;
  this->size = size;
  
  this->row = 0;
  this->col = 0;
  this->xb = 0;
  
  this->arrays = new Array*[size];
  for (int i=0; i<size; i++) {
    this->arrays[i] = new Array(block_id, i, x, w, y, params);
  }
}

Block::pim() {
}
