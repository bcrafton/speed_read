
#include "pim.h"

Block::Block(int block_id, int* x, int* w, int* y, Params* params) {
  this->block_id = block_id;
  this->params = params;
  
  this->row = 0;
  this->col = 0;
  this->xb = 0;
  
  this->arrays = new Array*[params->NBL];
  for (int i=0; i<params->NBL; i++) {
    this->arrays[i] = new Array(block_id, i, x, w, y, params);
  }
}

int Block::pim(int row) {
  for (int i=0; i<this->params->NBL; i++) {
    
  }
}
