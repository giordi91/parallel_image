#include <filters/convolution_filter.h>
#include <core/convolution.h>
#include <core/attribute.h>

void Convolution_filter::compute_serial(const uint8_t * source,
                        uint8_t* target)
{

	//make an instance of the filter
	convolution_serial(source, target,m_width,m_height,*st);

}
void Convolution_filter::compute_tbb(const uint8_t * source,
                        uint8_t* target)
{

	//make an instance of the filter
	convolution_tbb(source, target,m_width,m_height,*st);

}

void Convolution_filter::compute_cuda( uint8_t * source,
                        uint8_t* target)
{

	//make an instance of the filter
	convolution_cuda(source, target,m_width,m_height,*st);

}