#include <iostream>
#include <filters/bw_filter.h>
#include <tbb/parallel_for.h>

//cuda includes 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void Bw_filter::compute_serial(	const uint8_t * source,
                uint8_t* target)
{
	//instancing variablese
	int idx =0;
	uint8_t color;
	//looping the width
	for (int w=0; w<m_width; ++w )
	{
		//looping the height
		for (int h=0; h<m_height; ++h )
		{
			//computing index and color
			idx = (m_width*h)*3 + (w*3);
			color = uint8_t(0.21*float(source[idx])+0.72*float(source[idx+1]) + 0.07*float(source[idx+2]));
			//setting the color
			target[idx] = color;
			target[idx+1] = color;
			target[idx+2] = color;
		}
	}
}



void Bw_filter::compute_tbb(const uint8_t * source,
                			uint8_t* target)
{
	//create an instance of the class
	Apply_bw_tbb kernel(source,target,m_width,m_height);
	//kick the parallel for
	tbb::parallel_for(tbb::blocked_range2d<size_t>(0,m_width,0,m_height), kernel);


}

Apply_bw_tbb::Apply_bw_tbb(const uint8_t * source,
            uint8_t* target,
            const int &width,
            const int &height):m_source(source),m_target(target),
								m_width(width),m_height(height)
{

}

void Apply_bw_tbb::operator() (const tbb::blocked_range2d<size_t>& r)const
{
	//allocatin needed variables
	size_t idx =0;
	uint8_t color;

	for( size_t w=r.rows().begin(); w!=r.rows().end(); ++w ){
        for( size_t h=r.cols().begin(); h!=r.cols().end(); ++h ) 
           	{
				idx = (m_width*h)*3 + (size_t(w)*3);
				
				color = uint8_t(0.21*float(m_source[idx])+0.72*float(m_source[idx+1]) + 0.07*float(m_source[idx+2]));
				//setting the color value
				m_target[idx] = color;
				m_target[idx+1] = color;
				m_target[idx+2] = color;
			}
		}
}


//declaration of the kernel

void run_bw_kernel(const uint8_t *d_source, uint8_t *d_target, 
						const int width, const int height);

void Bw_filter::compute_cuda(const uint8_t * h_source,
                uint8_t* h_target)

{

	//calculating the size of the arrya
	int byte_size = m_width*m_height*3*(int)sizeof(uint8_t);

	//declaring gpu pointers
	uint8_t * d_source;
	uint8_t * d_target;

	//allocating memory on the gpu
	cudaMalloc((void **) &d_source,byte_size);
	cudaMalloc((void **) &d_target,byte_size);

	//copying memory to gpu
	cudaError_t s = cudaMemcpy(d_source, h_source, byte_size, cudaMemcpyHostToDevice);
	if (s != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(s));
	//run the kernel
	run_bw_kernel(d_source, d_target, m_width, m_height);
	
	//copying memory from gpu
	s = cudaMemcpy(h_target, d_target, byte_size, cudaMemcpyDeviceToHost);
	if (s != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(s));

	//freeing the memory
	cudaFree(d_source);
	cudaFree(d_target);
	

}

