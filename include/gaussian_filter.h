#include "core/stancil.h"
#include <cstdint>


#ifndef __PARALLEL_IMAGE_GAUSSIAN_STANCIL_H
#define __PARALLEL_IMAGE_GAUSSIAN_STANCIL_H 
/**
 * @brief stancil class for gaussian blur
 * @details This stancil generates a gaussian filter based on the standard
 * 			distribution
 * 
 */
class Gaussian_stancil: public GenericStancil
{
public:
	/**
	* @brief this is the constructor
	* 
	* @param sigma the sigma for the blur, you can think of this as the radius,
	* 				this will also directly controll the final size of the stancil,
	* 				
	* @param normalize wheter or not to normalize the final stancil
	*/
	Gaussian_stancil(const float sigma, 
					 const bool normalize);
	/**
	 * @brief This is the destructor of the class
	 */
	virtual ~Gaussian_stancil();
	
private:
	//internal sigma
	float m_sigma;
	//interal bool value
	bool m_normalize;

};

void gaussian_serial(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        const float sigma,
                        const bool normalize);


void gaussian_tbb(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        const float sigma,
                        const bool normalize
                        );


void gaussian_cuda(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        const float sigma,
                        const bool normalize
                        );

#endif
