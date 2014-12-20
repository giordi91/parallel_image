#include <string>

using namespace std;

#ifndef __PARALLEL_IMAGE_BITMAP__
#define __PARALLEL_IMAGE_BITMAP__

typedef struct __attribute__((__packed__))  
{
  int16_t  bfType;
  int32_t bfSize;
  int16_t  bfReserved1;
  int16_t  bfReserved2;
  int32_t bfOffBits;
} BITMAPFILEHEADER;


typedef struct __attribute__((__packed__))
{
    int     biSize;            // size of the structure
    int     biWidth;           // image width
    int     biHeight;          // image height
    int16_t      biPlanes;          // bitplanes
    int16_t      biBitCount;         // resolution 
    int32_t      biCompression;     // compression
    int32_t      biSizeImage;       // size of the image
    int     biXPelsPerMeter;   // pixels per meter X
    int     biYPelsPerMeter;   // pixels per meter Y
    int32_t      biClrUsed;         // colors used
    int32_t      biClrImportant;    // important colors

} BITMAPINFOHEADER;

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
    const BITMAPFILEHEADER* getFileHeader();
    const BITMAPINFOHEADER* getInfoHeader();
private:

	int m_width;
	int m_height;
	string m_bmp_type;
	int m_byte_size;
    BITMAPFILEHEADER m_bitmap_file_header;
    BITMAPINFOHEADER m_bitmap_info_header;
    

};

#endif