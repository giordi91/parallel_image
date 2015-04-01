#include "core/convolution.h" 
#include <iostream>
#include <algorithm> //min max

//tbb includes
#include <tbb/parallel_for.h>

//cuda includes 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

void convolution_serial(const uint8_t * source,
                        uint8_t* target,
                        const size_t &width,
                        const size_t &height,
                        const  Stancil &workStancil)
{

    int st_width = (int)workStancil.get_width();
    int st_height = (int)workStancil.get_height();
    int center_x = (int)(st_width/2);
    int center_y = (int)(st_height/2);

 
    int localX,localY,st_w,st_h,w,h;
    size_t idx,final_idx;

    float colorR,colorG,colorB;

    //should i do that?
    //matching type
    int int_height= (int)height;
    int int_width= (int)width;

    //looping for the width of the image
    for (h=0; h<int_height; ++h )
    {
        for (w=0; w<int_width; ++w )
        //looping the height of the image
        {
            //zeroing the value in memory
            colorR=0;
            colorG=0;
            colorB=0;
            //looping the stancil width
            idx = (width*h)*3 + (w*3);
            for (st_h=0; st_h< st_height; ++st_h)
            {
                localY = st_h - center_y;
                for (st_w=0; st_w< st_width; ++st_w)
                {
                    //compute relative position of the stancil pixel
                    localX = st_w - center_x;

                    //checking if we are in the image boundary
                    if ((localX +w  >= 0 && (localX + w < (int_width))) &&
                        (localY + h >= 0 && (localY+ h < (int_height))))

                    {
                        //if we reach here it means we are somewhere 
                        // in the filter and image which holds a valid positon

                        //now lets compute where in the buffer we are so we can get 
                        //the color and multiply by the filter value
                        //basic pixel
                        
                        //now we shift for the offsetted stencil positon
                        final_idx = idx + ((localX*3) + (localY*width*3));

                        colorR += float(source[final_idx])*(float)workStancil.get_value(st_w,st_h);
                        colorG += float(source[final_idx+1])*(float)workStancil.get_value(st_w,st_h);
                        colorB += float(source[final_idx+2])*(float)workStancil.get_value(st_w,st_h);

                    }//end boundary check
                }//end filter looping width

            }//end of filetr looping height
            target[idx] = (uint8_t)min(255.0f,max(0.0f,colorR));
            target[idx+1] = (uint8_t)min(255.0f,max(0.0f,colorG));
            target[idx+2] = (uint8_t)min(255.0f,max(0.0f,colorB));
        }//end of image looping width
    }//end image looping height
} 

void convolution_tbb(   const uint8_t * source,
                        uint8_t* target,
                        const size_t &width,
                        const size_t &height,
                        const  Stancil &workStancil)
{
    //create an instance of the class for blur parallel
    Apply_convolution_tbb kernel(source,target,width,height,workStancil);
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0,width,0,height), kernel);
   


}

Apply_convolution_tbb::Apply_convolution_tbb(const uint8_t * source,
                            uint8_t* target,
                            const size_t &width,
                            const size_t &height,
                            const  Stancil &workStancil):m_source(source),m_target(target),
                                m_width(width),m_height(height),
                                m_workStancil(&workStancil)
{

}

void Apply_convolution_tbb::operator() (const tbb::blocked_range2d<size_t>& r)const
{
    
    int st_width = (int)m_workStancil->get_width();
    int st_height = (int)m_workStancil->get_height();
    int center_x = (int)(st_width/2);
    int center_y = (int)(st_height/2);

 
    int localX,localY,st_w,st_h;
    size_t idx,final_idx;
 
    float colorR,colorG,colorB;

    //should i do that?
    //matching type
    int int_height= (int)m_height;
    int int_width= (int)m_width;

 	//extracting the ranges
 	int startRows = (int)r.rows().begin();
 	int endRows = (int)r.rows().end();
 	int startCols = (int)r.cols().begin();
 	int endCols = (int)r.cols().end();
 
    for( int w= startRows; w!=endRows; ++w ){
        for( int h=startCols; h!=endCols; ++h ) 
            {
                //zeroing the value in memory
                colorR=0;
                colorG=0;
                colorB=0;
                //looping the stancil width
                idx = (m_width*h)*3 + (w*3);

                for (st_h=0; st_h< st_height; ++st_h)
                {
                    localY = st_h - center_y;
                    for (st_w=0; st_w< st_width; ++st_w)
                    {
                        //compute relative position of the stancil pixel
                        localX = st_w - center_x;

                        //checking if we are in the image boundary
                        //better boundary checks here I am loosing some pixel on the edges
                        if (( (localX +w) >= 0 && (localX+w < (int_width))) &&
                            (localY+h >= 0 && (localY+h < (int_height))))

                        {
                            //if we reach here it means we are somewhere 
                            // in the filter and image which holds a valid positon

                            //now lets compute where in the buffer we are so we can get 
                            //the color and multiply by the filter value
                            //basic pixel
                            
                            //now we shift for the offsetted stencil positon
                            final_idx = idx + ((localX*3) + (localY*m_width*3));

                            colorR += float(m_source[final_idx])*m_workStancil->get_value(st_w,st_h);
                            colorG += float(m_source[final_idx+1])*m_workStancil->get_value(st_w,st_h);
                            colorB += float(m_source[final_idx+2])*m_workStancil->get_value(st_w,st_h);

                        }//end boundary check
                    }//end filter looping width

                }//end of filetr looping height

                m_target[idx] = (uint8_t)min(255.0f,max(0.0f,colorR));
	            m_target[idx+1] = (uint8_t)min(255.0f,max(0.0f,colorG));
	            m_target[idx+2] = (uint8_t)min(255.0f,max(0.0f,colorB));
            }
        }
}

void run_convolution_kernel( uint8_t *d_source, uint8_t *d_target, 
                        const size_t width, const size_t height,
                        const float * d_stancil,
                        const size_t st_width,
                        const size_t st_height);


void convolution_cuda( uint8_t * h_source,
                uint8_t* h_target,
                const size_t &width,
                const size_t &height,
                const  Stancil &workStancil)

{

    //the main buffers are already allocated, we just need 
    //to take care of the stancil allocation on the gpu
    size_t d_st_width = workStancil.get_width();
    size_t d_st_height= workStancil.get_height();
    size_t filter_byte_size = d_st_height*
                      d_st_width*
                      (size_t)sizeof(float);

    //declaring gpu pointers
    float * d_stancil; 

    //allocating memory on the gpu for stancil
    cudaMalloc((void **) &d_stancil,filter_byte_size);

    //copying memory to gpu
	const float * h_st_source = workStancil.get_data();
    cudaError_t s = cudaMemcpy(d_stancil, h_st_source, filter_byte_size, cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(s));

    //here run the kernel
    run_convolution_kernel(h_source,h_target,width,height,
    					d_stancil,d_st_width,d_st_height);

    //the memory of the main buffers is handled externally, means
    //we justneed to free the memory of the stancily we actually used
    cudaFree(d_stancil);


}
