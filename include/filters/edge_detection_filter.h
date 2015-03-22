#include "core/stancil.h"
#include "core/filter.h"
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
	Edge_detection_stancil(size_t sharpen_type = 0);
	/**
	 * @brief This is the destructor of the class
	 */
	virtual ~Edge_detection_stancil();
};

class Edge_detection_filter: public Filter
{
public:
    Edge_detection_filter(const int &width,
                const int &height,
                const int &detection_type);

    void compute_serial( const uint8_t * source,
                uint8_t* target);

    void compute_tbb(const uint8_t * source,
                uint8_t* target);
    void compute_cuda(const uint8_t * source,
                uint8_t* target);
private:
        Edge_detection_stancil  m_working_stancil;
        int m_detection_type;

};

#endif