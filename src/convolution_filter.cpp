#include "convolution_filter.h" 
#include <iostream>

//tbb includes
#include <tbb/parallel_for.h>

//cuda includes 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void convolution_serial(const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        const  Stancil &workStancil)
{

    int st_width = workStancil.get_width();
    int st_height = workStancil.get_height();
    int center_x = (int)(st_width/2.0);
    int center_y = (int)(st_height/2.0);


    int w,h,st_w,st_h,localX,localY,idx,final_idx;

    float colorR,colorG,colorB;
    //looping for the width of the image
    for (h=0; h<height; ++h )
    {
        for (w=0; w<width; ++w )
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
                    if ((localX +w  >= 0 && (localX + w < (width))) &&
                        (localY + h >= 0 && (localY+ h < (height))))

                    {
                        //if we reach here it means we are somewhere 
                        // in the filter and image which holds a valid positon

                        //now lets compute where in the buffer we are so we can get 
                        //the color and multiply by the filter value
                        //basic pixel
                        
                        //now we shift for the offsetted stencil positon
                        final_idx = idx + ((localX*3) + (localY*width*3));

                        colorR += float(source[final_idx])*workStancil.get_value(st_w,st_h);
                        colorG += float(source[final_idx+1])*workStancil.get_value(st_w,st_h);
                        colorB += float(source[final_idx+2])*workStancil.get_value(st_w,st_h);

                    }//end boundary check
                }//end filter looping width

            }//end of filetr looping height
            target[idx] = (uint8_t)colorR;
            target[idx+1] = (uint8_t)colorG;
            target[idx+2] = (uint8_t)colorB;
        }//end of image looping width
    }//end image looping height
}

void convolution_tbb(   const uint8_t * source,
                        uint8_t* target,
                        const int &width,
                        const int &height,
                        const  Stancil &workStancil)
{
    //create an instance of the class for blur parallel
    Apply_convolution_tbb kernel(source,target,width,height,workStancil);
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0,width,0,height), kernel);
   


}

Apply_convolution_tbb::Apply_convolution_tbb(const uint8_t * source,
                            uint8_t* target,
                            const int &width,
                            const int &height,
                            const  Stancil &workStancil):m_source(source),m_target(target),
                                m_width(width),m_height(height),
                                m_workStancil(&workStancil)
{

}

void Apply_convolution_tbb::operator() (const tbb::blocked_range2d<size_t>& r)const
{
    
    int st_width = m_workStancil->get_width();
    int st_height = m_workStancil->get_height();
    int center_x = (int)(st_width/2.0);
    int center_y = (int)(st_height/2.0);


    int st_w,st_h,localX,localY,idx,final_idx;

    float colorR,colorG,colorB;



    for( size_t w=r.rows().begin(); w!=r.rows().end(); ++w ){
        for( size_t h=r.cols().begin(); h!=r.cols().end(); ++h ) 
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
                        if (( (localX +w) >= 0 && (localX+w < (m_width))) &&
                            (localY+h >= 0 && (localY+h < (m_height))))

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
                m_target[idx] = (uint8_t)colorR;
                m_target[idx+1] = (uint8_t)colorG;
                m_target[idx+2] = (uint8_t)colorB;
            }
        }
}

void run_convolution_kernel( uint8_t *d_source, uint8_t *d_target, 
                        const int width, const int height,
                        const float * d_stancil,
                        const int st_width,
                        const int st_height);


void convolution_cuda(const uint8_t * h_source,
                uint8_t* h_target,
                const int &width,
                const int &height,
                const  Stancil &workStancil)

{
    //calculating the size of the arrya
    int byte_size = width*height*3*(int)sizeof(uint8_t);
    int filter_byte_size = workStancil.get_height()*
                      workStancil.get_width()*3*
                      (int)sizeof(float);
    int d_st_width = workStancil.get_width();
    int d_st_height= workStancil.get_height();



    //declaring gpu pointers
    uint8_t * d_source;
    uint8_t * d_target;
    float * d_stancil;

    //allocating memory on the gpu for source image,target,and stancil
    cudaMalloc((void **) &d_source,byte_size);
    cudaMalloc((void **) &d_target,byte_size);
    cudaMalloc((void **) &d_stancil,filter_byte_size);

    //copying memory to gpu
    cudaError_t s;
	s= cudaMemcpy(d_source, h_source, byte_size, cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(s));
    s = cudaMemcpy(d_stancil, workStancil.get_data(), filter_byte_size, cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(s));

    //here run
    run_convolution_kernel(d_source,d_target,width,height,
    					d_stancil,d_st_width,d_st_height);

    s = cudaMemcpy(h_target, d_target, byte_size, cudaMemcpyDeviceToHost);
    if (s != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(s));

    //freeing the memory
    cudaFree(d_source);
    cudaFree(d_target);
    cudaFree(d_stancil);


}
