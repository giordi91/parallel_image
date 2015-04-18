/** 
Abstract interface for the filters
"*/
#include <cstdint>
#include <vector>
#include <core/attribute.h>

#ifndef __PARALLEL_IMAGE_FILTER_H
#define __PARALLEL_IMAGE_FILTER_H 

class Filter
{
public:
	// virtual Filter(const int &width,
	// 		const int &height):m_width(width),m_height(height){};
	virtual void compute_serial( const uint8_t * source,
                					uint8_t* target)=0;
	virtual void compute_tbb(const uint8_t * source,
                					uint8_t* target)=0;
	virtual void compute_cuda( uint8_t * source,
                					uint8_t* target)=0;

	const std::vector<Attribute*> get_attributes()const
							{return m_attributes;};

	// virtual static Filter* create_filter(const int &width,
 //                				  	const int &height) =0;				


	virtual ~Filter(){};
protected:
	int m_width;
	int m_height;
	std::vector< Attribute*> m_attributes;


};

#endif