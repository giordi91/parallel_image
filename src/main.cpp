#include <iostream>
#include <fstream>
#include <cstring>
#include <bitmap.h>
#include <tbb/task_scheduler_init.h>
#include "tbb/tick_count.h"
#include <time.h>
#include <stdexcept>
#include <bw_filter.h>
#include <blur_filter.h>
#include <stancil.h>
#include <gaussian_stancil.h>
#include <convolution_filter.h>

using namespace std;

//TODO 
//MAKE SURE WE ARE DOING CONSEQUNETIAL ACCESS IN MEMORY (MOSTLY GPU),
//aka looping first hegiht then width

int main( int argc, char* argv[]) 
{
    cout<<"Initializing ..."<<endl;
    cout<<"Reading image...."<<std::endl;
    
    Bitmap testbmp;
    try
    {
        ///user_data/WORK_IN_PROGRESS/parallel_image/data/jessy.bmp
        testbmp.open("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessy.bmp");
    }
    catch(std::runtime_error &e)
    {
        std::cout<<e.what()<<endl;
        #if defined(WIN)
        system ("PAUSE");
        #endif
        return 0;
    }

    //gather the data
    unsigned int width = testbmp.get_width();
    unsigned int height = testbmp.get_height();
    unsigned int padded_size = testbmp.get_padded_size();
    Bitmap workingBmp(width, height, padded_size);
    std::cout<<"Image Info :"<<std::endl;
    std::cout<<"width: "<<width<<std::endl;
    std::cout<<"height: "<<height<<std::endl;

    //needed buffers
    uint8_t * src = testbmp.getRawData();
    uint8_t * target = workingBmp.getRawData();

    tbb::tick_count t0,t1;
     
    // blur test

    // int iterations = 19;
    // std::cout<<"Operation : 20 blur iterations \n"<<endl;
    // // //time the serial functon
    // std::cout<<"Running CPU serial..."<<endl; 
    // t0 = tbb::tick_count::now();
    // simple_blur_serial(src, target, width, height, iterations);
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" seconds \n" << endl;

    // std::cout<<"Running CPU parallel (TBB)..."<<endl; 
    // t0 = tbb::tick_count::now();
    // tbb::task_scheduler_init init(8);
    // //testing tbb
    // blur_tbb(src, target, width, height, iterations);
    // //terminating tbb
    // init.terminate();
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" seconds \n" << endl;

    // std::cout<<"Running GPU parallel (Cuda)..."<<endl; 
    // t0 = tbb::tick_count::now();
    // blur_cuda(src, target, width, height, iterations);
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" seconds" << endl;

    

    //  //BW test
    // tbb::tick_count t0,t1;
    // //time the serial functon

    // t0 = tbb::tick_count::now();
    // bw_serial(src, target, width, height);
    
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" s" << endl; 


    // t0 = tbb::tick_count::now();
    // tbb::task_scheduler_init init;
    // //testing tbb
    // bw_tbb(src, target, width, height);
    // //terminating tbb
    // init.terminate();
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" s" << endl; 


    // t0 = tbb::tick_count::now();
    // bw_cuda(src, target, width, height);
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" s" << endl; 
    

    //testing the stancil

    Gaussian_stancil st(15.0, true);
    // std::cout<<"gaussian_stancil values"<<std::endl;
    // st.log();

    // t0 = tbb::tick_count::now();
    // convolution_serial(src,target,width,height,st);
    // t1 = tbb::tick_count::now();
    // cout << "Computing SERIAL convolution"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 

    t0 = tbb::tick_count::now();
    convolution_tbb(src,target,width,height,st);
    t1 = tbb::tick_count::now();
    cout << "Computing parallel TBB convolution"<< endl;
    cout << (t1-t0).seconds()<<" s" << endl; 


    // t0 = tbb::tick_count::now();
    // convolution_cuda(src,target,width,height,st);
    // t1 = tbb::tick_count::now();
    // cout << "Computing parallel GPU convolution"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 


    try
    {   ///user_data/WORK_IN_PROGRESS/parallel_image/data/jessy.bmp
        workingBmp.save("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessyBW.bmp");
    }
    catch(std::runtime_error &e)
    {
        std::cout<<e.what()<<endl;
        system ("PAUSE");
        return 0;
    }

    #if defined(WIN)
    system ("PAUSE");
    #endif        
    return 0;

}