
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

void Layer::pim_sync() {
  int D = this->params->B / this->params->NWL;

  int* block_done = new int[this->params->B]();
  int* matrix_done = new int[D]();
  
  int done = 0;
  while (!done) {
    this->params->metrics[METRIC_CYCLE]++;

    for (int b=0; b<this->params->B; b++) {

      int row = this->row_map[b];
      int block_row = this->block_map[b];
            
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
        block_done[b] = 1;
        
        int group = b - (b % this->params->NWL);
        int d = b / this->params->NWL;
        
        int block_sync = 1;
        for (int i=group; i<group+this->params->NWL; i++) {
          block_sync = block_sync & block_done[i];
        }

        if (block_sync) {
          int next_row = this->row_queue[block_row];
          if (next_row < this->params->R) {
            for (int i=group; i<group+this->params->NWL; i++) {
              block_done[i] = 0;
              block_row = this->block_map[i];
              next_row = this->row_queue[block_row];
              this->row_map[i] = next_row;
              this->row_queue[block_row]++;
            }
          }
          else {
            matrix_done[d] = 1;
            int matrix_sync = 1;
            for (int i=0; i<D; i++) {
              matrix_sync = matrix_sync & matrix_done[i];
            }
            done = matrix_sync;
          }
        }
      }

    } // for (int b=0; b<this->params->B; b++) {
  } // while (!done) {
}








