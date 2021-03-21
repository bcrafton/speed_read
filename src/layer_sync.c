
#include "pim.h"

LayerSync::LayerSync(int* x, int* w, int* y, Params* params, int* block_map) {
  this->params = params;
  this->block_map = block_map;
  
  this->blocks = new Block*[params->B];
  for (int i=0; i<params->B; i++) {
    int block_id = block_map[i];
    this->blocks[i] = new Block(block_id, x, w, y, params); 
  }
  
  this->params->D = this->params->B / this->params->NWL;
  
  this->row_map = new int[this->params->D]();
  this->row_queue = 0;
  
  for (int i=0; i<this->params->D; i++) {
    this->row_map[i] = this->row_queue;
    this->row_queue++;
  }
}

void LayerSync::pim() {
  
  int* matrix_done = new int[this->params->D]();
  int* block_done = new int[this->params->B]();
  
  int done = 0;
  while (!done) {
    this->params->metrics[this->params->adc + METRIC_CYCLE]++;

    for (int d=0; d<this->params->D; d++) {
      for (int block_row=0; block_row<this->params->NWL; block_row++) {
        int b = d * this->params->NWL + block_row;
        if (block_done[b]) {
          this->params->metrics[this->params->adc + METRIC_STALL] += this->params->NBL;
        }
        else {
          this->params->metrics[this->params->adc + METRIC_BLOCK_CYCLE + block_row] += 1;
          block_done[b] = this->blocks[b]->pim(this->row_map[d]);
        }
      } 

      int block_sync = 1;
      for (int block_row=0; block_row<this->params->NWL; block_row++) {
        block_sync = block_sync & block_done[d * this->params->NWL + block_row];
      }

      if (block_sync) {
        int next_row = this->row_queue;
        if (next_row < this->params->R) {
          this->row_map[d] = next_row;
          this->row_queue++;
          
          for (int block_row=0; block_row<this->params->NWL; block_row++) {
            block_done[d * this->params->NWL + block_row] = 0;
          }
        }
        else {
          matrix_done[d] = 1;
        }
      }

    } // for (int d=0; d<D; d++) {
    
    done = 1;
    for (int i=0; i<this->params->D; i++) {
      done = done & matrix_done[i];
    }
    
  } // while (!done) {
}








