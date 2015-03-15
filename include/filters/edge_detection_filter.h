#include "core/stancil.h"
#include <cstdint>


#ifndef __PARALLEL_IMAGE_SHARPEN_FILTER_H
#define __PARALLEL_IMAGE_SHARPEN_FILTER_H 
/**
 * @brief stancil class for gaussian blur
 * @details This stancil generates a gaussian filter based on the standard
 * 			distribution
 * 
 */
class Edge_detection_stancil: public GenericStancil
{
public:
	/**
	* @brief this is the constructor

ril	*/
	Edge_detection_stancil(size_t sharpen_type = 0);
	/**
	 * @brief This is the destructor of the class
	 */
	virtual ~Edge_detection_stancil();
};
#endif


void edge_detection_serial(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        int detection_type);


void edge_detection_tbb(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        int detection_type
                        );


void edge_detection_cuda(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        int detection_type
                        );