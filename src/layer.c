
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

    done = 1;
    for (int b=0; b<this->params->B; b++) {
      
      done &= block_done[b];
      if (block_done[b]) continue;

      int row = this->row_map[b];
      int ret = this->blocks[b]->pim(row);
      
      if (ret) {
        int block_row = this->block_map[b];
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

