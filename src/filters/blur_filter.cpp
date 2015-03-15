#include <stdio.h>
#include <iostream>
#include <vector>
#include <filters/blur_filter.h>


//tbb includes
#include <tbb/parallel_for.h>


//cuda includes 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void simple_blur_serial(const uint8_t * source,
			            uint8_t* target,
			            const int &width,
			            const int &height,
			            const unsigned int iterations)
{
	//instancing needed variables
	std::vector<int> idxs;
	idxs.clear();
	idxs.resize(5,-1);
	int curr_id;
	float colorR,colorG,colorB,sum;

	//temp pointer used for the swap operation
	uint8_t * tmp;
	//dummy pointer used for the swap, this will hold the target pointer
	//so we dont override the target source pointer
	uint8_t * target_copy_ptr;
	uint8_t * sourc_copy_ptr;

	//generating a temp buffer we will use to copy the source,
	//in this way we can override it without compromise the 
	//src
	uint8_t * workingBuffer = new uint8_t[width*height*3];
	memcpy(workingBuffer, source, sizeof(uint8_t)*width*height*3);
	
	//assigning the dummy pointer  to a copy of the original pointer 
	sourc_copy_ptr = (uint8_t * )workingBuffer;
	target_copy_ptr=(uint8_t * )target;



	//looping for the number of iterations
	for (unsigned int iter =0; iter< iterations; ++iter)
	{
		//looping for the width
		for (int w=0; w<width; ++w )
		{
			//looping the height
			for (int h=0; h<height; ++h )
			{
				//central index
				idxs[0]=(width*h)*3 + (w*3);
				//left index
				idxs[1] = (w>0)?(idxs[0] -3) : -1;
				//up index
				idxs[2] = (h<(height-1))?((width*(h+1))*3) + (w*3) : -1;
				//right index
				idxs[3] = (w<width-1)?(idxs[0] +3) : -1;
				//top index
				idxs[4] = (h>1)?((width*(h-1))*3) + (w*3) : -1;

				//flushing colors and sum
				colorR=0.0;
				colorG=0.0;
				colorB=0.0;
				sum= 0.0;
				// looping for all the neighbours
				for( curr_id=0; curr_id<5; ++curr_id)
				{
					if (idxs[curr_id] != -1)
					{
						colorR += float(source[idxs[curr_id]]);
						colorG += float(source[idxs[curr_id]+1]);
						colorB += float(source[idxs[curr_id]+2]);
						++sum;
					}

				}	

				//setting the result of the average in the proper rgb
				target_copy_ptr[idxs[0]] = uint8_t (colorR/sum);
				target_copy_ptr[idxs[0]+1] = uint8_t (colorG/sum);
				target_copy_ptr[idxs[0]+2] = uint8_t (colorB/sum);
			}
		}

		//swapping the pointers for next iteration
		tmp = sourc_copy_ptr;
		sourc_copy_ptr = target_copy_ptr;
		target_copy_ptr = tmp;

	}
	//now at this point we are not sure if the pointer in the targer_copy_pointer
	//is the actual final target, it depends from how many time we spapped, aka
	// iteration is a even number the two pointer will match if it s odd, they 
	//dont , anyway if that s the case we do a mem copy in the final target buffer
	if (target_copy_ptr != target)
	{
		memcpy(target, target_copy_ptr, sizeof(uint8_t)*width*height*3);
	}

	//freeing previosuly allocated memory
	delete [] workingBuffer;
}			        

void blur_tbb(	uint8_t * source,
                uint8_t* target,
             	const int &width,
                const int &height,
                const unsigned int iterations)
{
	
	//dummy pointer used for the swap, this will hold the target pointer
	//so we dont override the target source pointer
	uint8_t * target_copy_ptr;
	uint8_t * sourc_copy_ptr;

	//generating a temp buffer we will use to copy the source,
	//in this way we can override it without compromise the 
	//src
	uint8_t * workingBuffer = new uint8_t[width*height*3];
	memcpy(workingBuffer, source, sizeof(uint8_t)*width*height*3);
	
	//assigning the dummy pointer  to a copy of the original pointer 
	sourc_copy_ptr = (uint8_t * )workingBuffer;
	target_copy_ptr= (uint8_t * )target;


	//create an instance of the class for blur parallel
	Apply_blur_tbb kernel(sourc_copy_ptr,target_copy_ptr,width,height,iterations);
	
	//looping for the iteration
	for (unsigned int iter =0; iter< iterations; ++iter)
	{
		//kick the parallel for
		tbb::parallel_for(tbb::blocked_range2d<size_t>(0,width,0,height), kernel);
		kernel.swap_pointers();
	}

	//now at this point we are not sure if the pointer in the targer_copy_pointer
	//is the actual final target, it depends from how many time we spapped, aka
	// iteration is a even number the two pointer will match if it s odd, they 
	//dont , anyway if that s the case we do a mem copy in the final target buffer
	if (target_copy_ptr != target)
	{
		memcpy(target, target_copy_ptr, sizeof(uint8_t)*width*height*3);
	}

	delete [] workingBuffer;


}

Apply_blur_tbb::Apply_blur_tbb(uint8_t * source,
            uint8_t* target,
            const int &width,
            const int &height,
            const unsigned int iterations):m_source(source),m_target(target),
								m_width(width),m_height(height),
								m_iterations(iterations)
{

}

void Apply_blur_tbb::operator() (const tbb::blocked_range2d<size_t>& r)const
{
	
	//instancing variablese
	std::vector<long unsigned int> idxs;
	idxs.clear();
	idxs.resize(5,-1);
	long unsigned int curr_id;
	float colorR,colorG,colorB,sum;

	for( size_t w=r.rows().begin(); w!=r.rows().end(); ++w ){
        for( size_t h=r.cols().begin(); h!=r.cols().end(); ++h ) 
           	{
	// //looping the given TBB range
	// for (long unsigned int w=r.begin(); w!=r.end(); ++w )
	// {
	// 	//looping the height
	// 	for (long unsigned int h=0; h<m_height; ++h )
	// 	{
			//central index
			idxs[0]=(m_width*h)*3 + (w*3);
			//left index
			idxs[1] = (w>0)?(idxs[0] -3) : -1;
			//up index
			idxs[2] = (h<(m_height-1))?((m_width*(h+1))*3) + (w*3) : -1;
			//right index
			idxs[3] = (w<m_width-1)?(idxs[0] +3) : -1;
			//bottom index
			idxs[4] = (h>1)?((m_width*(h-1))*3) + (w*3) : -1;

			//flushing the color and sum
			colorR=0.0;
			colorG=0.0;
			colorB=0.0;
			sum= 0.0;

			//looping the ids
			for( curr_id=0; curr_id<5; ++curr_id)
			{
				if (idxs[curr_id] != (long unsigned int)-1)
				{
					colorR += float(m_source[idxs[curr_id]]);
					colorG += float(m_source[idxs[curr_id]+1]);
					colorB += float(m_source[idxs[curr_id]+2]);
					++sum;
				}

			}	

			//setting the average result
			m_target[idxs[0]] = uint8_t (colorR/sum);
			m_target[idxs[0]+1] = uint8_t (colorG/sum);
			m_target[idxs[0]+2] = uint8_t (colorB/sum);
		}
	}
}


void Apply_blur_tbb::swap_pointers()
{
	//swapping the source target buffer
	uint8_t * tmp;
	tmp = m_source;
	m_source = m_target;
	m_target = tmp;
}

void run_blur_kernel( uint8_t *d_source, uint8_t *d_target, 
						const int width, const int height,
						const int iterations);

void blur_cuda(const uint8_t * h_source,
                uint8_t* h_target,
                const int &width,
                const int &height,
				const int iterations)
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
	run_blur_kernel(d_source, d_target, width, height, iterations);
	
	//copying memory from gpu
	s = cudaMemcpy(h_target, d_target, byte_size, cudaMemcpyDeviceToHost);
	if (s != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(s));

	//freeing the memory
	cudaFree(d_source);
	cudaFree(d_target);

}