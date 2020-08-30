
#include "pim.h"

Layer::Layer(int* x, int* w, int* y, Params* params, int* block_map) {
  this->params = params;
  this->block_map = block_map;
  
  this->blocks = new Block*[params->B];
  for (int i=0; i<params->B; i++) {
    int block_id = block_map[i];
    this->blocks[i] = new Block(block_id, x, w, y, params); 
  }
  
  this->row_map = new int[params->B]();
  this->row_queue = new int[params->NWL]();
  
  for (int i=0; i<params->B; i++) {
    int block_row = this->block_map[i];
    int next_row = this->row_queue[block_row];
    this->row_map[i] = next_row;
    this->row_queue[block_row]++;
  }
}

void Layer::pim() {
  int* block_done = new int[this->params->B]();
  
  int done = 0;
  while (!done) {
    this->params->metrics[METRIC_CYCLE]++;

    done = 1;
    for (int b=0; b<this->params->B; b++) {

      int row = this->row_map[b];
      int block_row = this->block_map[b];
      
      done &= block_done[b];
      
      // we add B*NBL incorrect stalls.
      // we add 1 incorrect cycle.
      if (block_done[b]) {
        this->params->metrics[METRIC_STALL] += this->params->NBL;
        continue;
      }
      else {
        this->params->metrics[METRIC_BLOCK_CYCLE + block_row] += 1;
      }

      int ret = this->blocks[b]->pim(row);
      if (ret) {
        int next_row = this->row_queue[block_row];
        if (next_row < this->params->R) {
          this->row_map[b] = next_row;
          this->row_queue[block_row]++;
        }
        else {
          block_done[b] = 1;
        }
      }
    } // for (int b=0; b<this->params->B; b++) {
  } // while (!done) {
}










