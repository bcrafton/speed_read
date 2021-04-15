
#include "pim.h"

Block::Block(int block_id, int* x, int* w, int* y, Params* params) {
  this->block_id = block_id;
  this->params = params;

  this->arrays = new Array*[params->NBL];
  for (int i=0; i<params->NBL; i++) {
    this->arrays[i] = new Array(block_id, i, x, w, y, params);
  }
}

int Block::pim(int row) {
  int done = 0;
  for (int i=0; i<this->params->NBL; i++) {
    this->arrays[i]->pim(row);
    this->arrays[i]->process(row);
    this->arrays[i]->collect(row);
    int ret = this->arrays[i]->update(row);
    if (i == 0) done = ret;
    else        assert(ret == done);
  }
  return done;
}
