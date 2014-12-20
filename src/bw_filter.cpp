#include <iostream>
#include <bw_filter.h>
#include <tbb/parallel_for.h>


void bw_serial(	const uint8_t * source,
                uint8_t* target,
             	const int &width,
                const int &height)
{

	//std::cout<<"serial black and white"<<std::endl;
	int idx =0;
	uint8_t color;
	for (int w=0; w<width; w++ )
	{
		for (int h=0; h<height; h++ )
		{
			idx = (width*h)*3 + (w*3);
			color = 0.21*float(source[idx]) +0.72*float(source[idx+1]) + 0.07*float(source[idx+2]);
			// 0.21 R + 0.72 G + 0.07 B
			target[idx] = color;
			target[idx+1] = color;
			target[idx+2] = color;

		}

	}
}

void bw_tbb(	const uint8_t * source,
                uint8_t* target,
             	const int &width,
                const int &height)
{
	Apply_bw_tbb kernel(source,target,width,height);
	tbb::parallel_for(tbb::blocked_range<size_t>(0,width), kernel);


}

Apply_bw_tbb::Apply_bw_tbb(const uint8_t * source,
            uint8_t* target,
            const int &width,
            const int &height):m_source(source),m_target(target),
								m_width(width),m_height(height)
{

}

void Apply_bw_tbb::operator() (const tbb::blocked_range<size_t>& r)const
{
	int idx =0;
	uint8_t color;
	for (int w=r.begin(); w!=r.end(); w++ )
	{
		for (int h=0; h<m_height; h++ )
		{
			idx = (m_width*h)*3 + (w*3);
			color = 0.21*float(m_source[idx]) +0.72*float(m_source[idx+1]) + 0.07*float(m_source[idx+2]);
			// 0.21 R + 0.72 G + 0.07 B
			m_target[idx] = color;
			m_target[idx+1] = color;
			m_target[idx+2] = color;

		}

	}

}