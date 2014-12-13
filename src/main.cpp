#include <iostream>
#include <fstream>
#include <cstring>
#include <bitmap.h>

using namespace std;

int main( int argc, char* argv[])
{
	cout<<"initializing ..."<<endl;
    Bitmap testbmp;
    testbmp.open("/home/giordi/WORK_IN_PROGRESS/parallel_image/data/tile.bmp");
    return 0;

}