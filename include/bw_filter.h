/** 
@file bw_filter.h
@brief Header for black and white filter.

This is the header containing the functions needed for doing the black and white

@author Marco Giordano
@bug No known bugs.
 */
#include <tbb/blocked_range.h>

/**
@brief this function performs a serial black and white filter
@param source: pointer to the source buffer
@param target: pointer to the targetr buffer
@param width: the width of the image
@param height: the height of the image
*/
void bw_serial(	const uint8_t * source,
                uint8_t* target,
                const int &width,
                const int &height);


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
	void operator() (const tbb::blocked_range<size_t>& r)const;

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