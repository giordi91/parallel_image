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
	 * @param grain_size : this is the base unit used to compute 
	 * the block size, and then based on that the grid size.
	 * Bewhare that the blocks are squared block of size grain_size*grain_size
	 * this mean the value you will input squared will need to be less
	 * of the max block size
	 *  
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

	/**
	 * @brief frees the memory allocated on the gpu for internal
	 *  buffers
	 */
	void free_internal_buffers();

	/**
	 * @brief get a device pointer to the source image
	 * @return uint8_t *
	 */
	uint8_t * get_source_buffer();

	/**
	 * @brief get a device pointer to the target image
	 * @return uint8_t *
	 */
	uint8_t * get_target_buffer();

	/**
	 * @brief getter for the grain size
	 * @return size_t
	 */
	size_t get_grain_size();

	/**
	 * @brief generic buffer allocator
	 * @details This function allocats a buffer on the device and
	 * returns a pointer to it, bewhare that the class has no ownership
	 * over that pointer, so is duty of the user to free it when everything
	 * is done
	 * 
	 * @param width the width of the needed buffer
	 * @param height the height of the needed buffer
	 * @param stride what's the strade of the buffer, in the case of
	 * an image which has just RGB the stide will be 3
	 * 
	 * @return uint8_t * device pointer
	 */
	uint8_t * allocate_device_buffer(size_t width,
									 size_t height,
									 size_t stride);
	/**
	 * @brief returns the block dim3 needed to kick kernels 
	 * @return dim 3
	 */
	dim3 get_block_dim();
	/**
	 * @brief returns the grid dim3 needed to kick kernels 
	 * @return dim 3
	 */
	dim3 get_grid_dim();


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