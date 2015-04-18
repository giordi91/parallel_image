#include "core/filter.h"
#include "core/stancil.h"
#include <cstdint>


#ifndef __PARALLEL_IMAGE_CONVOLUTION_FILTER_H
#define __PARALLEL_IMAGE_CONVOLUTION_FILTER_H 
/**
 * @brief This is a generic convolution class implementation
 * @details This class works by using a filter which is contained
 *          internally in the st pointer.
 *          The filter needs to be created in the update_data function,
 *          check it for more info
 * 
 */
class Convolution_filter: public Filter
{
public:
    /**
     * @brief This is the constructor
     * @details [long description]
     * 
     * @param width the width of the image to work on
     * @param height the height of the image to work on
     */
    Convolution_filter(const int &width,
                const int &height);

    /**
     * @brief The destructor
     */
    virtual ~Convolution_filter();

    /**
     * @brief Filter evaluation in serial
     * @details This function will evaluate the combination of filter
     *          and stancil using single cpu core computation
     * 
     * @param source the source image buffer
     * @param target the resulting image buffer
     */
    void compute_serial( const uint8_t * source,
                uint8_t* target);
    /**
     * @brief Filter evaluation in cpu parallel
     * @details This function will evaluate the combination of filter
     *          and stancil using parallel(Intel TBB)  core computation
     * 
     * @param source the source image buffer
     * @param target the resulting image buffer
     */
    void compute_tbb(const uint8_t * source,
                uint8_t* target);

     /**
     * @brief Filter evaluation in GPU
     * @details This function will evaluate the combination of filter
     *          and stancil using parallel(CUDA)  GPU  computation
     * 
     * @param source the source image buffer
     * @param target the resulting image buffer
     */
    void compute_cuda( uint8_t * source,
                uint8_t* target);

    /**
     * @brief Update internal data structure
     * @details This function is in charge of generating all the needed
     * data, for example this function will be used to generate the 
     * wanted filters based on the attributes of the class, since
     * the ui will hook up directly to the attribute class there wont
     * be need to pass the new values as arguments and we can use a generic
     * interface to regenarate the wanted stancil
     */
    virtual void update_data(){};

protected:
    //A pointer to the stancil
	Stancil * st;

};

#endif