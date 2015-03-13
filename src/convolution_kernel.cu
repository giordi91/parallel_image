
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include<iostream>
#include<stdio.h>


__global__ void blur_kernel(const uint8_t *d_source, uint8_t *d_target, 
                        const int width, const int height, const float *d_stancil)

{

    int idx =( (((blockIdx.y*blockDim.y) + threadIdx.y) *width) + 
                    (threadIdx.x + (blockIdx.x*blockDim.x))) *3;

    //setting the average result
    d_target[idxs[0]] = 255;
    d_target[idxs[0]+1] = 0;
    d_target[idxs[0]+2] = 0;

}



void run_convolution_kernel( uint8_t *d_source,  uint8_t *d_target, 
                        const int width, const int height, const  Stancil &workStancil)
{
    const int grainSize=16;
    int width_blocks,width_height;
    //computing the block size
    width_blocks = ((width%grainSize) != 0)?(width/grainSize) +1: (width/grainSize);
    width_height = ((height%grainSize) != 0)?(height/grainSize) +1: (height/grainSize);
    
    //setupping the block and grids
    const dim3 blockSize( grainSize, grainSize , 1); 
    const dim3 gridSize( width_blocks, width_height, 1); 

    //setupping clock
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //starting the clock
    cudaEventRecord(start);
    //kick the kernel
    uint8_t * tmp;
    for (int iter=0; iter<iterations; ++iter)
    {
        blur_kernel<<<gridSize, blockSize>>>(d_source, d_target, width,height,iterations);
        
        tmp = d_source;
        d_source = d_target;
        d_target = tmp;
    }
    //sincronizing device
    cudaDeviceSynchronize();

    //stop the clock
    cudaEventRecord(stop);

    //getting the time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<milliseconds<<std::endl;


    //checking for error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));


}