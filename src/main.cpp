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


using namespace std;

int main( int argc, char* argv[]) 
{
	cout<<"Initializing ..."<<endl;
	cout<<"Reading image...."<<std::endl;
	
    Bitmap testbmp;
    try
    {
    	testbmp.open("D:/PROGETTI_IN_CORSO/C/parallel_image/data/jessy.bmp");
	}
	catch(std::runtime_error &e)
	{
		std::cout<<e.what()<<endl;
		system ("PAUSE");
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

	
 	// blur test

 	int iterations = 19;
	std::cout<<"Operation : 20 blur iterations \n"<<endl;
	tbb::tick_count t0,t1;
	// //time the serial functon
	std::cout<<"Running CPU serial..."<<endl; 
	t0 = tbb::tick_count::now();
    simple_blur_serial(src, target, width, height, iterations);
    t1 = tbb::tick_count::now();
    cout << (t1-t0).seconds()<<" seconds \n" << endl;

	std::cout<<"Running CPU parallel (TBB)..."<<endl; 
	t0 = tbb::tick_count::now();
	tbb::task_scheduler_init init(8);
	//testing tbb
	blur_tbb(src, target, width, height, iterations);
	//terminating tbb
	init.terminate();
    t1 = tbb::tick_count::now();
    cout << (t1-t0).seconds()<<" seconds \n" << endl;
	
	std::cout<<"Running GPU parallel (Cuda)..."<<endl; 
	t0 = tbb::tick_count::now();
	blur_cuda(src, target, width, height, iterations);
    t1 = tbb::tick_count::now();
    cout << (t1-t0).seconds()<<" seconds" << endl;

	

	/*
    //BW test
	tbb::tick_count t0,t1;
	//time the serial functon

	t0 = tbb::tick_count::now();
	bw_serial(src, target, width, height);
    
	t1 = tbb::tick_count::now();
	cout << (t1-t0).seconds()<<" s" << endl; 


	t0 = tbb::tick_count::now();
	tbb::task_scheduler_init init(4);
	//testing tbb
	bw_tbb(src, target, width, height);
	//terminating tbb
	init.terminate();
	t1 = tbb::tick_count::now();
	cout << (t1-t0).seconds()<<" s" << endl; 


	t0 = tbb::tick_count::now();
	bw_cuda(src, target, width, height);
	t1 = tbb::tick_count::now();
	cout << (t1-t0).seconds()<<" s" << endl; 
	*/
    try
    {
    	workingBmp.save("D:/PROGETTI_IN_CORSO/C/parallel_image/data/jessyBlur.bmp");
    }
    catch(std::runtime_error &e)
	{
		std::cout<<e.what()<<endl;
		system ("PAUSE");
		return 0;
	}

	system ("PAUSE");

    return 0;

}