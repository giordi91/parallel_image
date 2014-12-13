#include <string>

using namespace std;

#ifndef __PARALLEL_IMAGE_BITMAP__
#define __PARALLEL_IMAGE_BITMAP__


class Bitmap
{

public:
    Bitmap();
    ~Bitmap();
    void open( const char* path);
    int width();
    int height();
    string type();
    int size();

private:

	int m_width;
	int m_height;
	string m_bmp_type;
	int m_byte_size;
    

};

#endif