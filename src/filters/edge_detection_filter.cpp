#include <filters/edge_detection_filter.h>
#include <core/convolution.h>

Edge_detection_stancil::Edge_detection_stancil(size_t sharpen_type)
{
	//setting the static size
	m_width = 3;
	m_height = 3;

	//allocating memory
	//the reason why we allocate on the heap is because this is inherited
	//from a generic class template which allow us to have arbitrary size
	//stancil
	m_data= new float[m_width*m_height];
	
	switch(sharpen_type)
	{
	
		case 1:
			m_data[0] = 1;
			m_data[1] = 0;
			m_data[2] = -1;

			//set row1
			m_data[3] = 0;
			m_data[4] = 0;
			m_data[5] = 0;

			//set row2
			m_data[6] = -1;
			m_data[7] = 0;
			m_data[8] = 1;
			break;

		case 2:
			m_data[0] = -1;
			m_data[1] = -1;
			m_data[2] = -1;

			//set row1
			m_data[3] = -1;
			m_data[4] = 8;
			m_data[5] = -1;

			//set row2
			m_data[6] = -1;
			m_data[7] = -1;
			m_data[8] = -1;
			break;

		
		default:
			m_data[0] = 0;
			m_data[1] = 1;
			m_data[2] = 0;

			//set row1
			m_data[3] = 1;
			m_data[4] = -4;
			m_data[5] = 1;

			//set row2
			m_data[6] = 0;
			m_data[7] = 1;
			m_data[8] = 0;


	}
}

Edge_detection_stancil::~Edge_detection_stancil()
{
	//no need to do anything, the virtual cascade of destructor is respected, means
	// the data will be free from the base class
}



void edge_detection_serial(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        int detection_type
                        )
{

	//make an instance of the filter
	Edge_detection_stancil st = Edge_detection_stancil(detection_type);
	convolution_serial(source, target,width,height,st);

}



void edge_detection_tbb(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        int detection_type
                        )
{

	//make an instance of the filter
	Edge_detection_stancil st = Edge_detection_stancil(detection_type);
	convolution_tbb(source, target,width,height,st);

}


void edge_detection_cuda(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        int detection_type
                        )
{

	//make an instance of the filter
	Edge_detection_stancil st = Edge_detection_stancil(detection_type);
	convolution_cuda(source, target,width,height,st);

}