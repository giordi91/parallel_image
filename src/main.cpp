#include <iostream>
#include <fstream>
#include <cstring>
#include <bitmap.h>
#include <bw_filter.h>

using namespace std;

int main( int argc, char* argv[])
{
	cout<<"initializing ..."<<endl;
    Bitmap testbmp;
    testbmp.open("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessy.bmp");

    //gather the data
    uint8_t * buffer = testbmp.getRawData();
    int width = testbmp.width();
    int height = testbmp.height();

    //allocate result buffer
	uint8_t *resultBuffer = new uint8_t[width*height*3];    

    bw_serial(buffer, resultBuffer, width, height);

    testbmp.setRawData(resultBuffer);

    testbmp.save("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessySaved.bmp");
    
    //cleanup 
    //no need to clean up since the bitmap class takes ownership of the pointer and 
    //cleans it up
    //delete [] resultBuffer;

    return 0;

}