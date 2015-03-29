
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h> 
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
    //
    if ((blockDim.y * blockIdx.y)+ threadIdx.y > (height-1))
    {
        return;
    }

    int x,y,localX,localY,final_idx,pixX,pixY, st_idx;
    //lets compute the coordinate of the pixel we are computing
    pixX = (blockIdx.x*blockDim.x) + threadIdx.x;
    pixY = (blockIdx.y*blockDim.y) + threadIdx.y;

    int idx = ((pixY *width) + (pixX)) *3;


    //computing the center of the filter
    int center_x = (int)(st_width/2.0);
    int center_y = (int)(st_height/2.0);
    
    //allocating/initializing  color variables
    float colorR = 0,colorG = 0,colorB = 0;

    //looping the height of the filter
    for (y=0; y<st_height; ++y)
    {

        localY = y - center_y;
        //looping the weidth of the filter
        for (x=0;x<st_width; ++x)
        {
            //lets compute where in the filter we are, computiing local
            //coordinate from the center
            localX = x - center_x;

            //boundary check 
            if (( (localX + pixX) >= 0 && ((localX+pixX) < width)) &&
                (localY+pixY >= 0 && ((localY+pixY) < height)))

            {

                //compute the final pixel to sample taking in to account 
                //the offset of the filter
                final_idx = idx + ((localX*3) + (localY*width*3));

                //compute the filter index buffer
                st_idx = x+ (y*st_width);

                colorR += float(d_source[final_idx])*d_stancil[st_idx];
                colorG += float(d_source[final_idx+1])*d_stancil[st_idx];
                colorB += float(d_source[final_idx+2])*d_stancil[st_idx];

            }//end of stencil boundary checking

        }//end of looping filter width

    }//end of looping filter height

    //setting the color to final buffer
    d_target[idx] = (uint8_t)min(255.0f,max(0.0f,colorR));
    d_target[idx+1] = (uint8_t)min(255.0f,max(0.0f,colorG));
    d_target[idx+2] = (uint8_t)min(255.0f,max(0.0f,colorB));

}



void run_convolution_kernel( uint8_t *d_source,  uint8_t *d_target, 
                        const size_t width, const size_t height, 
                        const  float *d_stancil,
                        const size_t st_width,
                        const size_t st_height)
{
    
	const int grainSize=16;
    int width_blocks,width_height;
    //computing the block size
    width_blocks = ((width%grainSize) != 0)?(width/grainSize) +1: (width/grainSize);
    width_height = ((height%grainSize) != 0)?(height/grainSize) +1: (height/grainSize);
    
    //setupping the block and grids
    const dim3 blockSize( grainSize, grainSize , 1); 
    const dim3 gridSize( width_blocks, width_height, 1); 

    //calling the actual kernel
    convolution_kernel<<<gridSize, blockSize>>>(d_source, 
                                                d_target, 
                                                width,
                                                height,
                                                d_stancil,
                                                st_width,
                                                st_height);
    
    //sincronizing device
    cudaDeviceSynchronize();

    //checking for error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

	
}