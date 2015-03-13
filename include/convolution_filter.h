#include <cstdint>
#include <stancil.h>
#include <tbb/blocked_range2d.h>

#ifndef __PARALLEL_IMAGE_CONVOLUTION_FILTER_H
#define __PARALLEL_IMAGE_CONVOLUTION_FILTER_H 


/**
@brief this function performs a serial convolution filter
@param source: pointer to the source buffer
@param target: pointer to the targetr buffer
@param width: the width of the image
@param height: the height of the image
@param workStancil: a pointer to the instance of 
                    the stancil we wish to use
*/
void convolution_serial(	const uint8_t * source,
			                uint8_t* target,
			                const int &width,
			                const int &height,
			                const  Stancil &workStancil);


void convolution_tbb(       const uint8_t * source,
                            uint8_t* target,
                            const int &width,
                            const int &height,
                            const  Stancil &workStancil);


class Apply_convolution_tbb 
{
public:
    Apply_convolution_tbb(const uint8_t * source,
                            uint8_t* target,
                            const int &width,
                            const int &height,
                            const  Stancil &workStancil);

    void operator() (const tbb::blocked_range2d<size_t>& r)const;

private:

    // internal pointer to the source buffer
    const uint8_t * m_source;
    // internal pointer to the target buffer
    uint8_t* m_target;
    //internal width of the image
    const int m_width;
    //internal height of the image
    const int m_height;
    //stancil pointer
    const Stancil *m_workStancil;


};


void convolution_cuda(const uint8_t * h_source,
                uint8_t* h_target,
                const int &width,
                const int &height,
                const  Stancil &workStancil);

#endif