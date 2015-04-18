#include <core/stancil.h>
#include <core/filter.h>
#include <filters/convolution_filter.h>
#include <cstdint>
#include <core/attribute.h>


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

class Gaussian_filter: public Convolution_filter
{
public:
    Gaussian_filter(const int &width,
                const int &height,
                const float &sigma=1.0f);

	AttributeTyped<float>m_sigma;

	void generate_filter();

    static Filter * create_filter(const int &width,
                          const int &height){ return new Gaussian_filter(width,height);}; 


};
#endif
