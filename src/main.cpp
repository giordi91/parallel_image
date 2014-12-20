#include <iostream>
#include <fstream>
#include <cstring>
#include <bitmap.h>
#include <bw_filter.h>
#include <tbb/task_scheduler_init.h>
#include "tbb/tick_count.h"
#include <time.h>

using namespace std;

int main( int argc, char* argv[])
{
	cout<<"initializing ..."<<endl;
    Bitmap testbmp;
    testbmp.open("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/emma.bmp");

    //gather the data
    uint8_t * buffer = testbmp.getRawData();
    int width = testbmp.width();
    int height = testbmp.height();

    //allocate result buffer
	uint8_t *resultBuffer = new uint8_t[width*height*3];    

	tbb::tick_count t0,t1;
	//time the serial functon

	t0 = tbb::tick_count::now();
    bw_serial(buffer, resultBuffer, width, height);
    
    t1 = tbb::tick_count::now();
    cout << (t1-t0).seconds()<<"S" << endl; 


    t0 = tbb::tick_count::now();
    tbb::task_scheduler_init init(4);
    //testing tbb
    bw_tbb(buffer, resultBuffer, width, height);
    //terminating tbb
    init.terminate();
    t1 = tbb::tick_count::now();
    cout << (t1-t0).seconds()<<"S" << endl; 
    testbmp.setRawData(resultBuffer);

    testbmp.save("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/emmaSave.bmp");
    
    //cleanup 
    //no need to clean up since the bitmap class takes ownership of the pointer and 
    //cleans it up
    //delete [] resultBuffer;

    return 0;

}