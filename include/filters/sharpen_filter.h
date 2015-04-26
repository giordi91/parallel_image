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
	/**
	 * @brief This is the constructor
	 * @param width: the width of the image to work on
	 * @param height: the height of the image to work on
	 */
    Sharpen_filter(const int &width,
                const int &height);

    /**
     * @brief This is the destructor
     */
    virtual ~Sharpen_filter(){};
    /**
     * @brief function triggered to update internal data
     * @details this function generates a filter based
     * on the m_detection_type value
     */
    virtual void update_data();

    /**
     * @brief Retunrs a string with the type of the filter
     * @return string
     */
    virtual std::string get_type();

    
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

};

#endif
