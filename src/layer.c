
#include "pim.h"

Layer::Layer(int* x, int* w, int* y, Params* params, int* block_map) {
  this->params = params;
  this->block_map = block_map;
  
  this->blocks = new Block*[params->B];
  for (int i=0; i<params->B; i++) {
    this->blocks[i] = new Block(i, params->NBL, x, w, y, params); 
  }
  
  this->row_map = new int[params->B];
  this->row_queue = new int[params->NWL];
}

void Layer::pim() {
  for (int b=0; b<this->params->B; b++) {
    int row = this->row_map[b];
    int done = this->blocks[b]->pim(row);
    if (done) {
      int wl = this->block_map[b];
      int next_row = this->row_queue[wl];
      this->row_queue[wl] += 1;
      this->row_map[b] = next_row;
    }
  }
  assert (0);
}
