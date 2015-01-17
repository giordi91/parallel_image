#include <iostream>
#include <bw_filter.h>
#include <tbb/parallel_for.h>

//cuda includes 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void bw_serial(	const uint8_t * source,
                uint8_t* target,
             	const int &width,
                const int &height)
{
	//instancing variablese
	int idx =0;
	uint8_t color;
	//looping the width
	for (int w=0; w<width; ++w )
	{
		//looping the height
		for (int h=0; h<height; ++h )
		{
			//computing index and color
			idx = (width*h)*3 + (w*3);
			color = uint8_t(0.21*float(source[idx])+0.72*float(source[idx+1]) + 0.07*float(source[idx+2]));
			//setting the color
			target[idx] = color;
			target[idx+1] = color;
			target[idx+2] = color;
		}
	}
}

void bw_tbb(	const uint8_t * source,
                uint8_t* target,
             	const int &width,
                const int &height)
{
	//create an instance of the class
	Apply_bw_tbb kernel(source,target,width,height);
	//kick the parallel for
	tbb::parallel_for(tbb::blocked_range2d<size_t>(0,width,0,height), kernel);


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
	int idx =0;
	uint8_t color;

	for( size_t i=r.rows().begin(); i!=r.rows().end(); ++i ){
        for( size_t j=r.cols().begin(); j!=r.cols().end(); ++j ) 
           	{
				//computing the index and the color
				// idx = (m_width*h)*3 + (int(w)*3);
				idx = (m_width*j)*3 + (int(i)*3);

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

void bw_cuda(const uint8_t * h_source,
                uint8_t* h_target,
                const int &width,
                const int &height)

{

	//calculating the size of the arrya
	int byte_size = width*height*3*(int)sizeof(uint8_t);

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
	run_bw_kernel(d_source, d_target, width, height);
	
	//copying memory from gpu
	s = cudaMemcpy(h_target, d_target, byte_size, cudaMemcpyDeviceToHost);
	if (s != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(s));

	//freeing the memory
	cudaFree(d_source);
	cudaFree(d_target);
	

}

