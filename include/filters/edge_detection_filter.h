#include <core/stancil.h>
#include <core/filter.h>
#include <filters/convolution_filter.h>
#include <cstdint>


#ifndef __PARALLEL_EDGE_DETECTION_FILTER_H
#define __PARALLEL_EDGE_DETECTION_FILTER_H
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
	Edge_detection_stancil(size_t detection_type = 0);
	/**
	 * @brief This is the destructor of the class
	 */
	virtual ~Edge_detection_stancil();
};

class Edge_detection_filter: public Convolution_filter
{
public:
    Edge_detection_filter(const int &width,
                const int &height,
                const size_t detection_type =0);

    size_t m_detection_type;
};

#endif