
#include "pim.h"

Params::Params(int R, int B, int C, int NWL, int NBL, int WL, int BL, int adc, float* adc_state, float* adc_thresh, float* lut_var, int* lut_rpr, long* metrics) {
  this->R = R;
  this->B = B;
  this->C = C;
  
  this->NWL = NWL;
  this->NBL = NBL;
  this->WL = WL;
  this->BL = BL;
  
  this->adc = adc;
  this->adc_state = adc_state;
  this->adc_thresh = adc_thresh;
  
  this->lut_var = lut_var;
  this->lut_rpr = lut_rpr;

  this->metrics = metrics;
}

