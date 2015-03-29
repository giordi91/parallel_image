#include <core/GPU_manager.h>
#include <iostream>
#include <stdexcept>      // std::invalid_argument
#include <string>

GPU_manager::GPU_manager(size_t width, size_t height,
						size_t grain_size):
						m_grain_size(grain_size),
						m_width(width), m_height(height),
						
						d_source(nullptr), d_target(nullptr)
{
	
	//first of all double check that the grain size is not too big
	//for the max per block threads
	struct cudaDeviceProp prop;
	cudaError_t result =  cudaGetDeviceProperties(&prop,0);	
	if (result != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(result));

    if ((m_grain_size*m_grain_size) > (unsigned int)prop.maxThreadsPerBlock)
    {
    	std::string error_msg = "GPU_manager::GPU_manager(): grain_size,";
    	error_msg += std::string("argument excede the max allowed block per size. \n");
    	error_msg += std::string("Check your card specs for more information");
    	throw std::invalid_argument(error_msg.c_str());
    }

    //computing the block size
    size_t width_blocks = ((m_width%m_grain_size) != 0)?(m_width/m_grain_size) +1: (m_width/m_grain_size);
    size_t width_height = ((m_height%m_grain_size) != 0)?(m_height/m_grain_size) +1: (m_height/m_grain_size);
    
    //setupping the block and grids
    m_kernel_block_size =  dim3( (unsigned int)m_grain_size, 
    							  (unsigned int)m_grain_size , 1); 
    m_kernel_grid_size= dim3( (unsigned int)width_blocks, 
    							(unsigned int)width_height, 1); 

	
    //allocate the needed buffers
   	d_source = allocate_device_buffer(m_width,m_height,3);
    d_target = allocate_device_buffer(m_width,m_height,3);

}

GPU_manager::~GPU_manager()
{
	free_internal_buffers();
}
size_t GPU_manager::get_width()
{
	return m_width;
}
size_t GPU_manager::get_height()
{
	return m_height;
}

void GPU_manager::free_internal_buffers()
{
	if (d_source)
	{
		cudaFree(d_source);
		d_source = nullptr;
	}

	if (d_target)
	{
		cudaFree(d_target);
		d_target = nullptr;
	}
}

uint8_t * GPU_manager::get_source_buffer()
{
	return d_source;
}
uint8_t * GPU_manager::get_target_buffer()
{
	return d_target;
}


uint8_t * GPU_manager::allocate_device_buffer(size_t width,
									 size_t height,
									 size_t stride)
{
	//calculating the size of the buffer
    size_t buffer_size = width*height*stride*(size_t)sizeof(uint8_t);

    uint8_t * d_buffer;
    //allocating memory on the gpu for source image,target
    cudaError_t result;
    result = cudaMalloc((void **) &d_buffer,buffer_size);
    if (result != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(result));

    return d_buffer;
}

size_t GPU_manager::get_grain_size()
{	
	return m_grain_size;
}