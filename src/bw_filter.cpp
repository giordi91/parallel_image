#include <iostream>
#include <bw_filter.h>

void bw_serial(	const uint8_t * source,
                uint8_t* target,
             	const int &width,
                const int &height)
{

	std::cout<<"serial black and white"<<std::endl;
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