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
                const int &height);

    void compute_serial( const uint8_t * source,
                uint8_t* target);

    void compute_tbb(const uint8_t * source,
                uint8_t* target);
    void compute_cuda( uint8_t * source,
                uint8_t* target);

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
                                  const int &height){ return new Bw_filter(width,height);}; 

    /**
     * @brief Implementation of the virtual function, does nothing
     */
    virtual void update_data(){};

    /**
     * @brief Retunrs a string with the type of the filter
     * @return string
     */
    virtual std::string get_type();   
};


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
#endif