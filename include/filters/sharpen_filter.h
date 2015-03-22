#include "core/stancil.h"
#include "core/filter.h"
#include <filters/convolution_filter.h>
#include <cstdint>


#ifndef __PARALLEL_IMAGE_SHARPEN_FILTER_H
#define __PARALLEL_IMAGE_SHARPEN_FILTER_H 
/**
 * @brief stancil class for gaussian blur
 * @details This stancil generates a gaussian filter based on the standard
 * 			distribution
 * 
 */
class Sharpen_stancil: public GenericStancil
{
public:
	/**
	* @brief this is the constructor
	* 
	* The filter is a 3x3 filter with values
	* 0  -1  0
	* -1  5  -1
	* 0  -1  0
	*/
	Sharpen_stancil();
	/**
	 * @brief This is the destructor of the class
	 */
	virtual ~Sharpen_stancil();
};

class Sharpen_filter: public Convolution_filter
{
public:
    Sharpen_filter(const int &width,
                const int &height);

    // void compute_serial( const uint8_t * source,
    //             uint8_t* target);

    // void compute_tbb(const uint8_t * source,
    //             uint8_t* target);
    // void compute_cuda(const uint8_t * source,
    //             uint8_t* target);
private:
	Sharpen_stancil  m_working_stancil;


};

#endif


// void sharpen_serial(const uint8_t * source,
//                         uint8_t* target,
//                         const int &width,
//                         const int &height);


// void sharpen_tbb(const uint8_t * source,
//                         uint8_t* target,
//                         const int &width,
//                         const int &height
//                         );


// void sharpen_cuda(const uint8_t * source,
//                         uint8_t* target,
//                         const int &width,
//                         const int &height
//                         );