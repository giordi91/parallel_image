#include <filters/gaussian_filter.h>
#include <core/convolution.h>
#include <iostream>		/* cout */
#include <math.h>       /* pow exp*/
const float  PI_F=3.14159265358979f;

Gaussian_stancil::Gaussian_stancil(	const float sigma, 
					 				const bool normalize):m_sigma(sigma),
														  m_normalize(normalize)
{
	//check that sigma is odd number 
	if ((size_t(m_sigma) % 2) == 0)
	{
		m_sigma +=1;
	}

	m_width = size_t(m_sigma)*2 +1;
	m_height = m_width;

	int width_int = (int)m_width ;
	int height_int = (int)m_height ;
	//allocating the buffer
	m_data= new float[m_width*m_height];

	//find the central coordiante of the map
	int center = width_int / 2;

	float localX, localY,toExp,res,total;
	total = 0;
	 
	float fistArg = 1.0f/(2.0f * PI_F * (float)pow(m_sigma,2.0f));

	for (int w=0; w<width_int; ++w)
	{

		for (int h=0; h<height_int; ++h)
		{
			//computing local coordinate
			localX = float(w-center);
			localY = float(h-center);

			//computing exponential part of the formula
			toExp = -((localX*localX) + (localY*localY))/(2.0f * m_sigma*m_sigma);
			//result
			res = fistArg *  (float)exp(toExp);
			//updating the total
			total += res;
			//updating the buffer
			m_data[w+ (h*m_width)] = res;
		}
	}	

	if (m_normalize == true)
	{
		for (size_t w=0; w<m_width; ++w)
		{
			for (size_t h=0; h<m_width; ++h)
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


Gaussian_filter::Gaussian_filter(const int &width,
                const int &height,
                const float &sigma):Convolution_filter(width,height), 
									m_sigma("sigma",sigma)
{
	update_data();
	m_attributes.push_back(&m_sigma);
}

void Gaussian_filter::update_data()
{
	if (st)
	{
		delete st;
	}
	st = new Gaussian_stancil(m_sigma.get_value(),1);
}


Filter * Gaussian_filter::create_filter(const int &width,
                          				const int &height)
{ 
	return new Gaussian_filter(width,height);
} 
