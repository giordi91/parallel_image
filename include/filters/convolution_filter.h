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

    virtual void update_data(){};

protected:
	Stancil * st;

};

#endif