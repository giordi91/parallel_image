#include <stancil.h>
#include <cstring>
#include <iostream>

GenericStancil::GenericStancil(): 	m_data(NULL), 
									m_width(0), 
									m_height(0)
{}

GenericStancil::GenericStancil(	const float * data, 
			 					const unsigned int width, 
								const unsigned int height): m_data(NULL), 
															m_width(width), 
															m_height(height)
{

	m_data = new float[m_width * m_height];
	memcpy(m_data, data, sizeof(float) * m_width * m_height);

}

float GenericStancil::get_value(unsigned int x, unsigned int y) const
{
	// TODO should i do boundaries check?
	// if i do trow exception and let it handle externally?
	return m_data[x+ (y*m_width)];
}

void GenericStancil::log()const
{
	for (unsigned int w=0; w< m_width; ++w)
	{
		for (unsigned int h=0; h<m_height; ++h)
		{
			std::cout<<get_value(w,h)<<" ";
		}
		std::cout<<std::endl;
	}

}

unsigned int GenericStancil::get_width()const
{
	return m_width;
}

unsigned int GenericStancil::get_height()const
{	
	return m_height;
}

unsigned int GenericStancil::get_size()const
{
	return (unsigned int)sizeof(float) * m_width * m_height;
}

const float * GenericStancil::get_data()const
{
	return m_data;
}

GenericStancil::~GenericStancil()
{
	if(m_data)
	{
		delete [] m_data;
	}
}