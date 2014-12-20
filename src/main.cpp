#include <iostream>
#include <fstream>
#include <cstring>
#include <bitmap.h>

using namespace std;

int main( int argc, char* argv[])
{
	cout<<"initializing ..."<<endl;
    Bitmap testbmp;
    testbmp.open("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessy.bmp");
    testbmp.save("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessySaved.bmp");
    return 0;

}