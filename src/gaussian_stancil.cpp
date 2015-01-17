#include <gaussian_stancil.h>
#include <iostream>

Gaussian_stancil::Gaussian_stancil(	const float sigma, 
					 				const bool normalize):m_sigma(sigma),
														  m_normalize(normalize)
{
	m_width = int(sigma)*2 +1;
	m_height = m_width; 

	m_data= new float[m_width*m_height];
}


Gaussian_stancil::~Gaussian_stancil()
{
	//no need to do anything, the virtual cascade of destructor is respected, means
	// the data will be freeed from the base class
}