
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include<iostream>
#include<stdio.h>


__global__ void convolution_kernel(const uint8_t *d_source, uint8_t *d_target, 
                        const int width, const int height, 
                        const float *d_stancil,
                        const int st_width,
                        const int st_height)

{

    //boundaries checking for width
    if ((blockDim.x * blockIdx.x)+ threadIdx.x > (width-1))
    {
        return;
    }
    if ((blockDim.y * blockIdx.y)+ threadIdx.y > (height-1))
    {
        return;
    }

    int idx =( (((blockIdx.y*blockDim.y) + threadIdx.y) *width) + 
                    (threadIdx.x + (blockIdx.x*blockDim.x))) *3;

    int x,y,localX,localY,final_idx;

    int center_x = (int)(st_width/2.0);
    int center_y = (int)(st_height/2.0);

    float colorR = 0;
    float colorG = 0;
    float colorB = 0;


    for (y=0; y<st_height; ++y)
    {
        for (x=0;x<st_width; ++x)
        {
            localX = x - center_x;
            localY = y - center_y;

            //boundary check 
            if ((localX >= 0 && (localX < (width))) ||
                            (localY >= 0 && (localY < (height))))

            {

                // final_idx = idx + ((localX*3) + (localY*m_width*3));
                final_idx = idx + ((localX*3) + (localY*width*3));

                colorR += float(d_source[final_idx])*d_stancil[x+ (y*st_width)];
                colorG += float(d_source[final_idx+1])*d_stancil[x+ (y*st_width)];
                colorB += float(d_source[final_idx+2])*d_stancil[x+ (y*st_width)];


            }

        }

    }

    //setting the average result
    d_target[idx] = (uint8_t)colorR;
    d_target[idx+1] = (uint8_t)colorG;
    d_target[idx+2] = (uint8_t)colorB;

}



void run_convolution_kernel( uint8_t *d_source,  uint8_t *d_target, 
                        const int width, const int height, 
                        const  float *d_stancil,
                        const int st_width,
                        const int st_height)
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
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //starting the clock
    //cudaEventRecord(start);
    //kick the kernel
    
    convolution_kernel<<<gridSize, blockSize>>>(d_source, d_target, width,height,
        d_stancil,st_width,st_height);
    
    //sincronizing device
    cudaDeviceSynchronize();

    //stop the clock
    //cudaEventRecord(stop);

    //getting the time
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //std::cout<<milliseconds<<std::endl;


    //checking for error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

	
}