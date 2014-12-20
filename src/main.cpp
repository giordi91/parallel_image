#include <iostream>
#include <fstream>
#include <cstring>
#include <bitmap.h>

using namespace std;

int main( int argc, char* argv[])
{
	cout<<"initializing ..."<<endl;
    Bitmap testbmp;
    testbmp.open("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/tile.bmp");

	const BITMAPINFOHEADER *header = testbmp.getInfoHeader();
	std::cout<<"getting width from pointer "<<header->biWidth<<endl;

    return 0;

}