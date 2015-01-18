#include <gaussian_stancil.h>
#include <iostream>		/* cout */
#include <math.h>       /* pow exp*/

const float  PI_F=3.14159265358979f;

Gaussian_stancil::Gaussian_stancil(	const float sigma, 
					 				const bool normalize):m_sigma(sigma),
														  m_normalize(normalize)
{
	//check that sigma is odd number 
	if ((int(m_sigma) % 2) == 0)
	{
		m_sigma +=1;
	}

	m_width = int(m_sigma)*2 +1;
	m_height = m_width;

	//allocating the buffer
	m_data= new float[m_width*m_height];

	//find the central coordiante of the map
	int center = m_width / 2;

	float localX, localY,toExp,res,total;
	total = 0;
	 
	float fistArg = 1.0f/(2.0f * PI_F * (float)pow(m_sigma,2.0f));
	for (int w=0; w<m_width; ++w)
	{

		for (int h=0; h<m_width; ++h)
		{
			//computing local coordinate
			localX = float(w-center);
			localY = float(h-center);
			//computing exponential part of the formula
			toExp = -((localX*localX) + (localY*localY))/(2.0f * m_sigma*m_sigma);
			//result
			res = fistArg *  exp(toExp);
			//updating the total
			total += res;
			//updating the buffer
			m_data[w+ (h*m_width)] = res;
		}
	}	

	if (m_normalize == true)
	{
		for (unsigned int w=0; w<m_width; ++w)
		{
			for (unsigned int h=0; h<m_width; ++h)
			{
				m_data[w+ (h*m_width)] /= total;
			}
		}
	}

}


Gaussian_stancil::~Gaussian_stancil()
{
	//no need to do anything, the virtual cascade of destructor is respected, means
	// the data will be freeed from the base class
}