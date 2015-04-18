/** 
Abstract interface for the filters
"*/
#include <cstdint>
#include <vector>
#include <core/attribute.h>

#ifndef __PARALLEL_IMAGE_FILTER_H
#define __PARALLEL_IMAGE_FILTER_H 

/**
 * @brief Abstarct filter class
 * @details This is a abstract filter class defining the interface
 *          for all the Filters
 * 
 */
class Filter
{
public:
	/**
     * @brief Abstract Filter evaluation in serial
     * @details This function will evaluate the combination of filter
     *          and stancil using single cpu core computation
     * 
     * @param source the source image buffer
     * @param target the resulting image buffer
     */
	virtual void compute_serial( const uint8_t * source,
                					uint8_t* target)=0;
	/**
     * @brief Abstract Filter evaluation in cpu parallel
     * @details This function will evaluate the combination of filter
     *          and stancil using parallel(Intel TBB)  core computation
     * 
     * @param source the source image buffer
     * @param target the resulting image buffer
     */
	virtual void compute_tbb(const uint8_t * source,
                					uint8_t* target)=0;
	/**
     * @brief Abstract Filter evaluation in GPU
     * @details This function will evaluate the combination of filter
     *          and stancil using parallel(CUDA)  GPU  computation
     * 
     * @param source the source image buffer
     * @param target the resulting image buffer
     */
	virtual void compute_cuda( uint8_t * source,
                					uint8_t* target)=0;

	/**
	 * @brief Returns a list of the class attributes
	 * @return the vector holding pointers to the attributes
	 */
	const std::vector<Attribute*> get_attributes()const
							{return m_attributes;};

	/**
     * @brief create the needed filter based on the class attributes
     * @details This function needs to be reimplemented from each convolution
     * filter, the class will generate the filter accordingly based
     * on the Attributes of the class, in this way the UI can connect
     * directly to the attributes and manipulate its value and after
     * that can call a generic generate_filter function withouth
     * need of apssing arguments
     */
    virtual void update_data()=0; 			

    /**
     * @brief The destructor
     */
	virtual ~Filter(){};
protected:
	//Width of the image to work on
	int m_width;
	//Height of the image to work on
	int m_height;
	//Array holding pointers to the class attributes
	std::vector< Attribute*> m_attributes;


};

#endif