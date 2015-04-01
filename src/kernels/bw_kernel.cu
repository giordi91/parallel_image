
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include<iostream>
#include<stdio.h>

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
	/*
	//compute the index up to the block
	//this is the expanded computation of the index easier to understand
	//and explained
	// ((blockIdx.y*blockDim.y)  this computes how many rows we need to skip
								//based on the blocks at which we sum the rows
								 //to skip internally at the block itself (threadIdx.y)
	//   which gives us the total rows to skip, all this is multiplicated by 
	// the width of the picture
	int rowsToSkip = ((blockIdx.y*blockDim.y) + threadIdx.y) *width;

	// now the last part of the index to compute is the partial row of the thread
	//first we take into consideration  the blocks 
	//(blockIdx.x*blockDim.x)
	//then we add the internal poistion in the block  threadIdx.x
	//we get this 
	int localRowIndex = threadIdx.x + (blockIdx.x*blockDim.x);
	//to compute the final index we add those two indexes giving the final index,
	//since our data is RGB we multiply everyhing by 3, since each pixel is 3 uint8_t
	//values
	//and we get
	int idx = (rowsToSkip + localRowIndex) *3;
	*/
	//this is the compressed version of the index calculation
	int idx = ( (((blockIdx.y*blockDim.y) + threadIdx.y) *width) + 
					(threadIdx.x + (blockIdx.x*blockDim.x))) *3;
	//int idx = (blockIdx.y*blockDim.y*width*3);
	//int localIdx = (( (threadIdx.x + (blockIdx.x*blockDim.x))) +((width) * threadIdx.y))*3;
	//idx += localIdx;
	uint8_t color = uint8_t(0.21*float(d_source[idx])+0.72*float(d_source[idx+1]) + 0.07*float(d_source[idx+2]));
	d_target[idx] = color;
	d_target[idx+1] = color;
	d_target[idx+2] = color; 
}
  
void run_bw_kernel(const uint8_t *d_source,  uint8_t *d_target, 
						const int width, const int height)
{
	const int grainSize=16;
	int width_blocks,width_height;
	//computing the block size
	width_blocks = ((width%grainSize) != 0)?(width/grainSize) +1: (width/grainSize);
	width_height = ((height%grainSize) != 0)?(height/grainSize) +1: (height/grainSize);
	
	//setupping the block and grids
	const dim3 blockSize( grainSize, grainSize , 1); 
	const dim3 gridSize( width_blocks, width_height, 1); 

	//kick the kernel
	bw_kernel<<<gridSize, blockSize>>>(d_source, d_target, width,height);

	//sincronizing device
	cudaDeviceSynchronize();
	
	//checking for error
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));


}
