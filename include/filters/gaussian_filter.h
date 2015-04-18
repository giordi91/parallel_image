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


	/**
     * @brief Static fucntion for generating class instance
     * @details This function is used by the factoty class (Filter_manager)
     *          for generating on the fly and on request the wanted filter
     * 
     * @param width: the width of the image to work on
     * @param height: the height of the image to work on
     * 
     * @return A pointer to a live istance of the class
     */
    static Filter * create_filter(const int &width,
                          		  const int &height); 
    /**
     * @brief function triggered to update internal data
     * @details this function generates a filter based
     * on the m_detection_type value
     */
	virtual void update_data();

public:
	//The attribute for the sigma of the gaussian function
	AttributeTyped<float>m_sigma;

};
#endif
