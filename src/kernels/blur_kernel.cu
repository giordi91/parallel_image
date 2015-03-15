
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include<iostream>
#include<stdio.h>

__global__ void blur_kernel(const uint8_t *d_source, uint8_t *d_target, 
						const int width, const int height, const int iterations)
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
	int idxs[5] = {-1,-1,-1,-1,-1};
	//central pixel
	//local row index
	int localWidth = threadIdx.x + (blockIdx.x*blockDim.x);
	int localHeight= threadIdx.y + (blockIdx.y*blockDim.y);

	idxs[0] =( (((blockIdx.y*blockDim.y) + threadIdx.y) *width) + 
					(threadIdx.x + (blockIdx.x*blockDim.x))) *3;
	//left pixel, if not on the edge we move back one pixel (aka 3 colors)
	idxs[1] = (localWidth>0)?(idxs[0] -3) : -1;
	//up index, if we are not out of the height we add one full row att the original pixel
	idxs[2] = (localHeight<(height-1))? (idxs[0] + (width*3)) : -1;
	//right index, if we are not on the edge we add one pixel (aka 3 colors)
	idxs[3] = (localWidth<width-1)?(idxs[0] +3) : -1;
	//bottom index, if not on the bottom we remove a full row
	idxs[4] = (localHeight>1)? (idxs[0] - (width*3)) : -1;


	float colorR,colorG,colorB;
	float sum;
	//looping the ids
	for( int curr_id=0; curr_id<5; ++curr_id)
	{
		if (idxs[curr_id] != (long unsigned int)-1)
		{
			colorR += float(d_source[idxs[curr_id]]);
			colorG += float(d_source[idxs[curr_id]+1]);
			colorB += float(d_source[idxs[curr_id]+2]);
			++sum;
		}
	}	
	//setting the average result
	d_target[idxs[0]] = uint8_t (colorR/sum);
	d_target[idxs[0]+1] = uint8_t (colorG/sum);
	d_target[idxs[0]+2] = uint8_t (colorB/sum);
}


void run_blur_kernel( uint8_t *d_source,  uint8_t *d_target, 
						const int width, const int height, const int iterations)
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
