#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstddef> 
#include <cstdint>

#ifndef __PARALLEL_IMAGE_GPU_MANAGER_H__
#define __PARALLEL_IMAGE_GPU_MANAGER_H__


class GPU_manager
{
public:
	/**
	 * @brief this is the constructor
	 * @param width : the width of the image we will work on
	 * @param height : the height of the image we will work on
	 */
	GPU_manager( size_t width, size_t height, size_t grain_size =16);

	~GPU_manager();

	/**
	 * @brief getter function for the image width
	 * @return size_t, the width of the image
	 */
	size_t get_width();
	
	/**
	 * @brief getter function for the image height
	 * @return size_t, the height of the image
	 */
	size_t get_height();

	void free_internal_buffers();

	uint8_t * get_source_buffer();
	uint8_t * get_target_buffer();


private:
	//the grain size used to make square blocks
	size_t m_grain_size;
	//the dim3 block size used to plug directly in the 
	//kernels calls
	dim3 m_kernel_block_size;
	//the dim3 defining the grid size for the blocks
	dim3 m_kernel_grid_size;
	//the width of the image to work on
	size_t m_width;
	//the height of thei mage to work on
	size_t m_height;

	//internal gpu pointers
	//source device buffer
	uint8_t * d_source;
	//target device buffer
    uint8_t * d_target;
};

#endif