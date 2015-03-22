/** 
Abstract interface for the filters
"*/
#include <cstdint>

#ifndef __PARALLEL_IMAGE_FILTER_H
#define __PARALLEL_IMAGE_FILTER_H 


class Filter
{
public:
	Filter(const int &width,
			const int &height):m_width(width),m_height(height){};
	virtual void compute_serial( const uint8_t * source,
                					uint8_t* target)=0;
	virtual void compute_tbb(const uint8_t * source,
                					uint8_t* target)=0;
	virtual void compute_cuda(const uint8_t * source,
                					uint8_t* target)=0;
protected:
	int m_width;
	int m_height;
};

#endif