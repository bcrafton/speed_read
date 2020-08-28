
#include "pim.h"

Block::Block(int size, int* x, int* w, int* y, Params* params) {
  this->size = size;
  
  this->row = 0;
  this->col = 0;
  this->xb = 0;
  
  this->arrays = new Array*[size];
  for (int i=0; i<size; i++) {
    this->arrays[i] = new Array(i, x, w, y, params);
  }
}


