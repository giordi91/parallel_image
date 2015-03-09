#include <cstdint>
#include <stancil.h>


#ifndef __PARALLEL_IMAGE_CONVOLUTION_FILTER_H
#define __PARALLEL_IMAGE_CONVOLUTION_FILTER_H 


/**
@brief this function performs a serial convolution filter
@param source: pointer to the source buffer
@param target: pointer to the targetr buffer
@param width: the width of the image
@param height: the height of the image
*/
void convolution_serial(	const uint8_t * source,
			                uint8_t* target,
			                const int &width,
			                const int &height,
			                const * stancil);

#endif