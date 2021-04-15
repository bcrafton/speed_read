
#include "pim.h"

Params::Params(int R, int B, int C, int NWL, int NBL, int WL, int BL, int adc, int max_rpr, float* adc_state, float* adc_thresh, float* lut_var, int* lut_rpr, int* lut_bias, long* metrics, int sync, int method, int skip, int ABFT, int ABFT_XB, int ABFT_BL) {
  this->R = R;
  this->B = B;
  this->C = C;
  
  this->NWL = NWL;
  this->NBL = NBL;
  this->WL = WL;
  this->BL = BL;
  
  this->adc = adc;
  this->max_rpr = max_rpr;

  this->adc_state = adc_state;
  this->adc_thresh = adc_thresh;
  
  this->lut_var = lut_var;
  this->lut_rpr = lut_rpr;
  this->lut_bias = lut_bias;

  this->metrics = metrics;
  
  this->sync = sync;
  this->method = method;
  this->skip = skip;
  
  this->ABFT    = ABFT;
  this->ABFT_XB = ABFT_XB;
  this->ABFT_BL = ABFT_BL;

  this->XB            = 8 + ABFT_XB;
  this->COL_PER_ADC   = 8;
  this->TOTAL_ADC     = 32 + ABFT_BL;

  this->XB_data = this->XB - this->ABFT_XB;
  this->BL_data = this->TOTAL_ADC - this->ABFT_BL;
}


