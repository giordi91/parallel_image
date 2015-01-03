/** 
@file blur_filter.h
@brief Header filer for the blur_filter.

This is the header containing the functions needed for doing the 
blur filter 

@author Marco Giordano
@bug No known bugs.
 */
#include <tbb/blocked_range.h>

/**
@brief this function performs a serial average blur filter
@param source: pointer to the source buffer
@param target: pointer to the targetr buffer
@param width: the width of the image
@param height: the height of the image
@param height: how many blur iterations to do
*/
void simple_blur_serial(const uint8_t * source,
			            uint8_t* target,
			            const int &width,
			            const int &height,
			            const unsigned int iterations
			            );



/**
@brief this function performs a parallel TBB based blur filter
@param source: pointer to the source buffer
@param target: pointer to the targetr buffer
@param width: the width of the image
@param height: the height of the image
@param height: how many blur iterations to do
*/
void blur_tbb(uint8_t * source,
            uint8_t* target,
            const int &width,
            const int &height,
            const unsigned int iterations);



/**
@brief This is the class used to kick the parallel TBB run
*/
class Apply_blur_tbb
{
public:
    /**
    @brief this is the constructor
    @param source: pointer to the source buffer
    @param target: pointer to the targetr buffer
    @param width: the width of the image
    @param height: the height of the image
    @param height: how many blur iterations to do
    */
	Apply_blur_tbb(uint8_t * source,
            uint8_t* target,
            const int &width,
            const int &height,
            const unsigned int iterations);

    /**
    @brief the () operator called by TBB
    @param r: the range the thread had to work on
    */
	void operator() (const tbb::blocked_range<size_t>& r)const;
    void swap_pointers();
private:
    // internal pointer to the source buffer
	uint8_t * m_source;
    // internal pointer to the target buffer
    uint8_t* m_target;
    //internal width of the image
    const long unsigned int m_width;
    //internal height of the image
    const long unsigned int m_height;
    //internal number of iterations;
    const unsigned int m_iterations;
};



void blur_cuda(const uint8_t * h_source,
                uint8_t* h_target,
                const int &width,
                const int &height,
				const int iterations);