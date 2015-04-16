#include "core/filter.h"
#include "core/stancil.h"
#include <cstdint>


#ifndef __PARALLEL_IMAGE_CONVOLUTION_FILTER_H
#define __PARALLEL_IMAGE_CONVOLUTION_FILTER_H 

class Convolution_filter: public Filter
{
public:
    Convolution_filter(const int &width,
                const int &height);

    virtual ~Convolution_filter();

    void compute_serial( const uint8_t * source,
                uint8_t* target);

    void compute_tbb(const uint8_t * source,
                uint8_t* target);
    void compute_cuda( uint8_t * source,
                uint8_t* target);

    /**
     * @brief create the needed filter based on the class attributes
     * @details This function needs to be reimplemented from each convolution
     * filter, the class will generate the filter accordingly based
     * on the Attributes of the class, in this way the UI can connect
     * directly to the attributes and manipulate its value and after
     * that can call a generic generate_filter function withouth
     * need of apssing arguments
     */
    virtual void generate_filter()=0; 

protected:
	Stancil * st;

};

#endif