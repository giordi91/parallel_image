#include <core/stancil.h>
#include <core/filter.h>
#include <filters/convolution_filter.h>
#include <cstdint>
#include <core/attribute.h>

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

    /**
     * @brief Retunrs a string with the type of the filter
     * @return string
     */
    virtual std::string get_type();   
public:
	/**
	 * the attribute defining the type of detection filter
	 */
    AttributeTyped<size_t> m_detection_type;


};

#endif