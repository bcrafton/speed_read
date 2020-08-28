
#include "pim.h"

Layer::Layer(int size, int* x, int* w, int* y, Params* params, int* block_map) {
  this->size = size;
  this->blocks = new Block*[size];
  this->block_map = block_map;
  for (int i=0; i<size; i++) {
    this->blocks[i] = new Block(i, params->NBL, x, w, y, params); 
  }
}

void Layer::pim() {
  assert (0);
}
