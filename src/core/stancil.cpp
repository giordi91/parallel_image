#include <core/stancil.h>
#include <cstring>

#include <iostream>

GenericStancil::GenericStancil(): 	m_data(NULL), 
									m_width(0), 
									m_height(0)
{}

GenericStancil::GenericStancil(	const float * data, 
			 					const size_t width, 
								const size_t height): m_data(NULL), 
															m_width(width), 
															m_height(height)
{

	m_data = new float[m_width * m_height];
	memcpy(m_data, data, sizeof(float) * m_width * m_height);

}

inline float GenericStancil::get_value(size_t x, size_t y) const
{
	// TODO should i do boundaries check?
	// if i do trow exception and let it handle externally?
	return m_data[x+ (y*m_width)];
}

void GenericStancil::log()const
{
	for (size_t w=0; w< m_width; ++w)
	{
		for (size_t h=0; h<m_height; ++h)
		{
			std::cout<<get_value(w,h)<<" ";
		}
		std::cout<<std::endl;
	}

}

size_t GenericStancil::get_width()const
{
	return m_width;
}

size_t GenericStancil::get_height()const
{	
	return m_height;
}

size_t GenericStancil::get_size()const
{
	return (size_t)sizeof(float) * m_width * m_height;
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