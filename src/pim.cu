
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define DLLEXPORT extern "C"

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(char *x, char *w, char *y, int X, int Z, int B, int NWL, int WL, int RPR, int RPR_MAX)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if( (row >= X) || (col >= Z) ) return;
  
  int x_sum = 0;
  int y_sum = 0;
  for (int xb=0; xb<B; xb++) {
    for (int wb=0; wb<B; wb++) {
      for (int nwl=0; nwl<NWL; nwl++) {
        x_sum = 0;
        y_sum = 0;
        for (int wl=0; wl<WL; wl++) {
          int x_addr = row*NWL*WL*B + nwl*WL*B + wl*B + xb;
          int w_addr = nwl*WL*Z*B + wl*Z*B+ col*B + wb;

          x_sum += x[x_addr];
          y_sum += x[x_addr] & w[w_addr];

          int flag = (x_sum == RPR) || (wl == (WL - 1));
          int y_addr = row*Z*B*B*NWL*RPR_MAX + col*B*B*NWL*RPR_MAX + xb*B*NWL*RPR_MAX + wb*NWL*RPR_MAX + nwl*RPR_MAX + y_sum;
          y[y_addr] += flag;
          y_sum = (!flag) * y_sum;
          assert (y_sum <= RPR);
          x_sum = (x_sum % RPR);
        }
      }
    }
  }
} 

DLLEXPORT int pim(char* x, char* w, char* y, int X, int Z, int B, int NWL, int WL, int RPR, int RPR_MAX)
{
    // allocate memory in host RAM, h_cc is used to store CPU result
    char *h_x, *h_w, *h_y;
    cudaMallocHost((void **) &h_x, sizeof(char)*X*NWL*WL*B);
    cudaMallocHost((void **) &h_w, sizeof(char)*NWL*WL*Z*B);
    cudaMallocHost((void **) &h_y, sizeof(char)*X*Z*B*B*NWL*RPR_MAX);
    memcpy(h_x, x, sizeof(char)*X*NWL*WL*B);
    memcpy(h_w, w, sizeof(char)*NWL*WL*Z*B);

    // allocate memory space on the device 
    char *d_x, *d_w, *d_y;
    cudaMalloc((void **) &d_x, sizeof(char)*X*NWL*WL*B);
    cudaMalloc((void **) &d_w, sizeof(char)*NWL*WL*Z*B);
    cudaMalloc((void **) &d_y, sizeof(char)*X*Z*B*B*NWL*RPR_MAX);
    cudaMemcpy(d_x, h_x, sizeof(char)*X*NWL*WL*B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, sizeof(char)*NWL*WL*Z*B, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (X + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (Z + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_x, d_w, d_y, X, Z, B, NWL, WL, RPR, RPR_MAX);    

    // transfer results from device to host
    cudaMemcpy(h_y, d_y, sizeof(char)*X*Z*B*B*NWL*RPR_MAX, cudaMemcpyDeviceToHost);
    memcpy(y, h_y, sizeof(char)*X*Z*B*B*NWL*RPR_MAX);

    cudaThreadSynchronize();

    /////////////////////////////////////////////////////

    // free memory
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);
    cudaFreeHost(h_x);
    cudaFreeHost(h_w);
    cudaFreeHost(h_y);
    return 0;
}




