#include <filters/sharpen_filter.h>
#include <iostream>		/* cout */
#include <math.h>       /* pow exp*/
#include <filters/convolution_filter.h>

Sharpen_stancil::Sharpen_stancil()
{
	//setting the static size
	m_width = 3;
	m_height = 3;

	//allocating memory
	//the reason why we allocate on the heap is because this is inherited
	//from a generic class template which allow us to have arbitrary size
	//stancil
	m_data= new float[m_width*m_height];
	//set row 0
	m_data[0] = 0;
	m_data[1] = -1;
	m_data[2] = 0;

	//set row1
	m_data[3] = -1;
	m_data[4] = 5;
	m_data[5] = -1;

	//set row2
	m_data[6] = 0;
	m_data[7] = -1;
	m_data[8] = 0;
}

Sharpen_stancil::~Sharpen_stancil()
{
	//no need to do anything, the virtual cascade of destructor is respected, means
	// the data will be free from the base class
}



void sharpen_serial(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height
                        )
{

	//make an instance of the filter
	Sharpen_stancil st = Sharpen_stancil();
	convolution_serial(source, target,width,height,st);

}



void sharpen_tbb(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height
                        )
{

	//make an instance of the filter
	Sharpen_stancil st = Sharpen_stancil();
	convolution_tbb(source, target,width,height,st);

}


void sharpen_cuda(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height
                        )
{

	//make an instance of the filter
	Sharpen_stancil st = Sharpen_stancil();
	convolution_cuda(source, target,width,height,st);

}