
#include "pim.h"

Array::Array(int block_id, int array_id, int* x, int* w, int* y, Params* params) {  
  this->block_id = block_id;
  this->array_id = array_id;

  this->x = x;
  this->w = w;
  this->y = y;
  
  this->params = params;

  this->xb = 0;
  this->col = 0;
  this->wl_ptr = 0;
  this->wl_sum = 0;
  this->pdot = new int[VECTOR_SIZE]();

  this->sum_XB       = new int[VECTOR_SIZE]();
  this->checksum_XB  = new int[VECTOR_SIZE]();
  this->sum_ADC      = new int[VECTOR_SIZE]();
  this->checksum_ADC = new int[VECTOR_SIZE]();
}

int Array::clear() {
}

int Array::pim(int row) {
  assert(row < this->params->R);

  if (this->params->skip) {
    return this->pim_skip(row);
  }
  else {
    return this->pim_base(row);
  }
}

int Array::pim_skip(int row) {
  memset(this->pdot, 0, sizeof(int) * VECTOR_SIZE);
  // int rpr = this->params->lut_rpr[this->xb * 8 + this->col];
  int rpr = this->params->adc;

  int xaddr_ROW = (row * this->params->NWL * this->params->WL * this->params->XB);
  int xaddr_NWL = (this->block_id * this->params->WL * this->params->XB);
  int xaddr_WL  = (this->wl_ptr * this->params->XB);
  int xaddr_XB  = xb;
  int xaddr     = xaddr_ROW + xaddr_NWL + xaddr_WL + xaddr_XB;
  assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));
  
  while ((this->wl_ptr < this->params->WL) && ((this->wl_sum + this->x[xaddr]) <= rpr)) {
    assert((this->x[xaddr] == 0) || (this->x[xaddr] == 1));
    
    if (this->x[xaddr]) {
      this->wl_sum += 1;
      
      for (int adc_ptr=0; adc_ptr<this->params->TOTAL_ADC; adc_ptr++) {
        int bl_ptr = adc_ptr * this->params->COL_PER_ADC + col;
        int waddr_NWL = (this->block_id * this->params->WL * this->params->NBL * this->params->BL);
        int waddr_WL  = (this->wl_ptr * this->params->NBL * this->params->BL);
        int waddr_NBL = (this->array_id * this->params->BL);
        int waddr_BL  = bl_ptr;
        int waddr     = waddr_NWL + waddr_WL + waddr_NBL + waddr_BL;
        this->pdot[bl_ptr] += this->w[waddr];
      }
    }
    
    this->wl_ptr += 1;
    // careful with placement of this, has to come after wl_ptr update.
    xaddr_ROW = (row * this->params->NWL * this->params->WL * this->params->XB);
    xaddr_NWL = (this->block_id * this->params->WL * this->params->XB);
    xaddr_WL  = (this->wl_ptr * this->params->XB);
    xaddr_XB  = xb;
    xaddr     = xaddr_ROW + xaddr_NWL + xaddr_WL + xaddr_XB;
  }
}

int Array::pim_base(int row) {
}

int Array::process(int row) {
  // int rpr = this->params->lut_rpr[this->xb * 8 + this->col];
  int rpr = this->params->adc;

  for (int adc_ptr=0; adc_ptr<this->params->TOTAL_ADC; adc_ptr++) {
    int bl_ptr = adc_ptr * this->params->COL_PER_ADC + col;
    int wb = col;

    int key = rand() % 1001;
    int var_addr = this->pdot[bl_ptr] * 1001 + key;
    float var = this->params->lut_var[var_addr];
    float pdot_var = this->pdot[bl_ptr] + var;
    
    int pdot_adc;
    if ((pdot_var > 0.20) && (pdot_var < 1.00)) {
      pdot_adc = 1;
    }
    else {
      pdot_adc = min(max((int) round(pdot_var), 0), min(this->params->adc, rpr));
    }

    int xb_flag = xb      < this->params->XB_data;
    int bl_flag = adc_ptr < this->params->ADC_data;

    if (bl_flag & xb_flag) {
      int c = (bl_ptr + this->array_id * this->params->BL_data) / 8; // 8 = COL / ADC ... I think.
      int yaddr = row * this->params->C + c;
      int shift = wb + xb;
      int sign = (wb == 7) ? (-1) : 1;
      this->y[yaddr] += sign * (pdot_adc << shift);
    }

    if (bl_flag) {
      this->sum_ADC[this->xb] += pdot_adc;
    }
    else {
      int scale = pow(2, adc_ptr - this->params->ADC_data);
      this->checksum_ADC[this->xb] += pdot_adc * scale; 
    }

    if (xb_flag) { 
      this->sum_XB[adc_ptr] += pdot_adc;
    }
    else { 
      int scale = pow(2, xb - this->params->XB_data);
      this->checksum_XB[adc_ptr] += pdot_adc * scale; 
    }

  } // for (int adc_ptr=0; adc_ptr<this->params->BL; adc_ptr+=8) {
}

int Array::collect(int row) {
}

int Array::correct(int row) {
}

int Array::correct_static(int row) {
}

/*
int Array::ABFT(int row) {
  if (this->wl_ptr == this->params->WL && this->xb == (this->params->XB - 1)) {
    for (int bit=0; bit<this->params->XB; bit++) {
      int MOD = pow(2, this->params->ABFT_ADC);
      int checksum1 = this->sum_ADC[bit]      % MOD;
      int checksum2 = this->checksum_ADC[bit] % MOD;
      assert (checksum1 == checksum2);
    }
    for (int bit=0; bit<this->params->TOTAL_ADC; bit++) {
      int MOD = pow(2, this->params->ABFT_XB);
      int checksum1 = this->sum_XB[bit]      % MOD;
      int checksum2 = this->checksum_XB[bit] % MOD;
      assert (checksum1 == checksum2);
    }
    memset(this->sum_ADC,      0, sizeof(int) * VECTOR_SIZE);
    memset(this->checksum_ADC, 0, sizeof(int) * VECTOR_SIZE);
    memset(this->sum_XB,       0, sizeof(int) * VECTOR_SIZE);
    memset(this->checksum_XB,  0, sizeof(int) * VECTOR_SIZE);
  }
}
*/

int compute_error(int A, int B, int MOD) {
  int error;
  if      (A == (MOD - 1) && B == 0) { error = -1;    }
  else if (B == (MOD - 1) && A == 0) { error = 1;     }
  else                               { error = A - B; }
  return error;
}

int Array::ABFT(int row) {
  if (this->wl_ptr == this->params->WL && this->xb == (this->params->XB - 1)) {
    int flag = 0;
    for (int bit1=0; bit1<this->params->XB; bit1++) {
      for (int bit2=0; bit2<this->params->TOTAL_ADC; bit2++) {

        int MOD_ADC = pow(2, this->params->ABFT_ADC);
        int adc_sum1 = this->sum_ADC[bit1] % MOD_ADC;
        int adc_sum2 = this->checksum_ADC[bit1] % MOD_ADC;
        int adc_error = compute_error(adc_sum2, adc_sum1, MOD_ADC);

        int MOD_XB = pow(2, this->params->ABFT_XB);
        int xb_sum1 = this->sum_XB[bit2] % MOD_XB;
        int xb_sum2 = this->checksum_XB[bit2] % MOD_XB;
        int xb_error = compute_error(xb_sum2, xb_sum1, MOD_XB);
        
        if (adc_error != 0 && xb_error != 0) {
          flag = 1;
          printf ("(%d %d) XB~(%d %d) ADC~(%d %d) (%d %d) ", bit1, bit2, xb_sum1, xb_sum2, adc_sum1, adc_sum2, xb_error, adc_error);
        }
      }
    }
    memset(this->sum_ADC,      0, sizeof(int) * VECTOR_SIZE);
    memset(this->checksum_ADC, 0, sizeof(int) * VECTOR_SIZE);
    memset(this->sum_XB,       0, sizeof(int) * VECTOR_SIZE);
    memset(this->checksum_XB,  0, sizeof(int) * VECTOR_SIZE);
    if (flag) printf("\n");
  }
}

int Array::update(int row) {
  this->wl_sum = 0;
  if (this->wl_ptr == this->params->WL) {
    this->wl_ptr = 0;
    this->xb = (this->xb + 1) % this->params->XB;
    if (this->xb == 0) {
      this->col = (this->col + 1) % 8;
      if (this->col == 0) {
        return 1;
      }
    }
  }
  return 0;
}




