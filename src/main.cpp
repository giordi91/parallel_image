#include <iostream>
#include <fstream>
#include <cstring>
#include <bitmap.h>
#include <bw_filter.h>
#include <tbb/task_scheduler_init.h>
#include "tbb/tick_count.h"
#include <time.h>
#include <stdexcept>

using namespace std;

int main( int argc, char* argv[])
{
	cout<<"initializing ..."<<endl;
    Bitmap testbmp;
    try
    {
    	testbmp.open("/home/giordi/WORK_IN_PROGRESS/parallel_image/data/jessy.bmp");
	}
	catch(std::runtime_error &e)
	{
		std::cout<<e.what()<<endl;
		return 0;
	}

	//gather the data
    uint width = testbmp.get_width();
    uint height = testbmp.get_height();
    uint padded_size = testbmp.get_padded_size();
    Bitmap workingBmp(width, height, padded_size);

    //needed buffers
    uint8_t * src = testbmp.getRawData();
    uint8_t * target = workingBmp.getRawData();
 

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

    try
    {
    	workingBmp.save("/home/giordi/WORK_IN_PROGRESS/parallel_image/data/jessyBW.bmp");
    }
    catch(std::runtime_error &e)
	{
		std::cout<<e.what()<<endl;
		return 0;
	}


    return 0;

}