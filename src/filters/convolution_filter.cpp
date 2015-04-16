#include <filters/convolution_filter.h>
#include <core/convolution.h>
#include <core/attribute.h>

Convolution_filter::Convolution_filter(const int &width,
                						const int &height):st(nullptr)
{
        m_width = width;
        m_height= height;
        m_attributes.clear();
}

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


Convolution_filter::~Convolution_filter()
{
	if(st)
	{
		delete st;
	}
}