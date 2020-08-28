
#include "pim.h"

Block::Block(int block_id, int* x, int* w, int* y, Params* params) {
  this->block_id = block_id;
  this->params = params;
  
  this->col = 0;
  this->xb = 0;
  
  this->arrays = new Array*[params->NBL];
  for (int i=0; i<params->NBL; i++) {
    this->arrays[i] = new Array(block_id, i, x, w, y, params);
  }
}

int Block::pim(int row) {
  int rpr_addr = this->xb * 8 + this->col;
  int rpr = this->params->lut_rpr[rpr_addr];
  assert(rpr_addr >= 0);
  assert(rpr_addr < 64); 
  assert(rpr > 0);
  assert(rpr <= 64);

  int done = 0;
  for (int i=0; i<this->params->NBL; i++) {
    int ret = this->arrays[i]->pim(row, this->col, this->xb, rpr);
    if (i == 0) done = ret;
    else        assert(ret == done);
  } 
  
  if (done) {
    this->col = (this->col + 1) % 8;
    if (this->col == 0) {
      this->xb = (this->xb + 1) % 8;
      if (this->xb == 0) {
        return 1;
      }
    }
  }
  return 0;
}
