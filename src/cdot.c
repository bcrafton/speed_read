
#include <stdio.h>

// make sure (bl <= 1024), malloc would be too slow.
// if we just pick a size large enough we will be okay
#define VECTOR_SIZE 1024
int pdot[VECTOR_SIZE]; 

int pim_kernel(int* x, int* w, int wl, int bl, int* y)
{

    int ncol = bl / 8;

    int psum = 0;
    int wl_ptr = 0;
        
    for (int bl_ptr=0; bl_ptr<bl; bl_ptr++) { pdot[bl_ptr] = 0; }
    for (int col=0; col<ncol; col++)        { y[col] = 0; }
    
    while (wl_ptr < wl) {
        int wl_sum = 0;
        for (int bl_ptr=0; bl_ptr<bl; bl_ptr++) { pdot[bl_ptr] = 0; }
        
        while ((wl_ptr < wl) && (wl_sum + x[wl_ptr] <= 8)) {
            if (x[wl_ptr]) {
                wl_sum += 1;
                for (int bl_ptr=0; bl_ptr<bl; bl_ptr++) {
                    pdot[bl_ptr] += w[wl_ptr * bl + bl_ptr];
                }
            }
            wl_ptr += 1;
        }
        psum += 1;

        for (int col=0; col<ncol; col++) {
            for (int wb=0; wb<8; wb++) {
                y[col] += (pdot[col * 8 + wb] << wb);
            }
            y[col] -= wl_sum * 128;
        }
    }
    return psum;
}
