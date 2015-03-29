#include <core/GPU_manager.h>
#include <iostream>

GPU_manager::GPU_manager(size_t width, size_t height,
						size_t grain_size):
						m_grain_size(grain_size),
						m_width(width), m_height(height),
						
						d_source(nullptr), d_target(nullptr)
{
	
    //computing the block size
    size_t width_blocks = ((m_width%m_grain_size) != 0)?(m_width/m_grain_size) +1: (m_width/m_grain_size);
    size_t width_height = ((m_height%m_grain_size) != 0)?(m_height/m_grain_size) +1: (m_height/m_grain_size);
    
    //setupping the block and grids
    m_kernel_block_size =  dim3( (unsigned int)m_grain_size, 
    							  (unsigned int)m_grain_size , 1); 
    m_kernel_grid_size= dim3( (unsigned int)width_blocks, 
    							(unsigned int)width_height, 1); 

	
	//calculating the size of the buffer
    size_t buffer_size = width*height*3*(size_t)sizeof(uint8_t);
    
    //allocating memory on the gpu for source image,target
    cudaError_t result;
    result = cudaMalloc((void **) &d_source,buffer_size);
    if (result != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(result));
    result = cudaMalloc((void **) &d_target,buffer_size);

    if (result != cudaSuccess) 
        printf("GPU_manager::GPU_Manager Error allocating initial buffers: %s\n",
        		 cudaGetErrorString(result));

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