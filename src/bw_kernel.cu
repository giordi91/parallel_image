
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>


#include<stdlib.h>
#include <stdint.h>
#include<iostream>
#include<stdio.h>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void bw_kernel(const uint8_t *d_source, uint8_t *d_target, 
						const int width, const int height)
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

	//compute the index up to the block
	int idx = (blockIdx.y*blockDim.y*width*3);
	int localIdx = (((threadIdx.x*3) + blockIdx.x*blockDim.x*3) +((width*3) * threadIdx.y));
	idx += localIdx;
	uint8_t color = uint8_t(0.21*float(d_source[idx])+0.72*float(d_source[idx+1]) + 0.07*float(d_source[idx+2]));
	d_target[idx] = color;
	d_target[idx+1] = color;
	d_target[idx+2] = color; 
}
  
void run_bw_kernel(const uint8_t *d_source,  uint8_t *d_target, 
						const int width, const int height)
{
	int grainSize=16;
	int width_blocks,width_height;
	//computing the block size
	width_blocks = ((width%grainSize) != 0)?(width/grainSize) +1: (width/grainSize);
	width_height = ((height%grainSize) != 0)?(height/grainSize) +1: (height/grainSize);

	std::cout<<width_blocks<<std::endl;
	std::cout<<width_height<<std::endl;

	std::cout<<((width_blocks*width_height*grainSize*grainSize)-(width*height))<<std::endl;
	const dim3 blockSize( grainSize, grainSize , 1); 
	const dim3 gridSize( width_blocks, width_height, 1); 

	//setupping clock

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	bw_kernel<<<gridSize, blockSize>>>(d_source, d_target, width,height);
	cudaEventRecord(stop);

	//getting the time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaDeviceSynchronize();
	//std::cout<<"kernel execution time in ms: "<<(milliseconds)<<std::endl;	
	
	//checking for error
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));


}