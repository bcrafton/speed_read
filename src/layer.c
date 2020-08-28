
#include "pim.h"

Layer::Layer(int size, int* x, int* w, int* y, Params* params) {
  this->size = size;
  this->blocks = new Block*[size];
  for (int i=0; i<size; i++) {
    this->blocks[i] = new Block(i, params->NBL, x, w, y, params); 
  }
}

void Layer::pim() {
}
