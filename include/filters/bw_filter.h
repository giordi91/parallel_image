/** 
@file bw_filter.h
@brief Header for black and white filter.

This is the header containing the functions needed for doing the black and white

@author Marco Giordano
@bug No known bugs.
 */
#include <core/filter.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>


#ifndef __PARALLEL_IMAGE_BW_FILTER_H
#define __PARALLEL_IMAGE_BW_FILTER_H 

/**
@brief this function performs a serial black and white filter
@param source: pointer to the source buffer
@param target: pointer to the targetr buffer
@param width: the width of the image
@param height: the height of the image
*/

class Bw_filter: public Filter
{
public:
    Bw_filter(const int &width,
                const int &height):Filter(width,height){};

    void compute_serial( const uint8_t * source,
                uint8_t* target);

    void compute_tbb(const uint8_t * source,
                uint8_t* target);
    void compute_cuda(const uint8_t * source,
                uint8_t* target);


};

// void bw_serial(	const uint8_t * source,
//                 uint8_t* target,
//                 const int &width,
//                 const int &height);


/**
@brief this function performs a parallel TBB based black and white filter
@param source: pointer to the source buffer
@param target: pointer to the targetr buffer
@param width: the width of the image
@param height: the height of the image
*/
void bw_tbb(const uint8_t * source,
            uint8_t* target,
            const int &width,
            const int &height);

/**
@brief This is the class used to kick the parallel TBB run
*/
class Apply_bw_tbb
{
public:
    /**
    @brief this is the constructor
    @param source: pointer to the source buffer
    @param target: pointer to the targetr buffer
    @param width: the width of the image
    @param height: the height of the image
    */
	Apply_bw_tbb(const uint8_t * source,
            uint8_t* target,
            const int &width,
            const int &height);

    /**
    @brief the () operator called by TBB
    @param r: the range the thread had to work on
    */
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
};


void bw_cuda(const uint8_t * h_source,
                uint8_t* h_target,
                const int &width,
                const int &height);

#endif