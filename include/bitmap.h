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
    int         biSize;            // size of the structure
    int         biWidth;           // image width
    int         biHeight;          // image height
    int16_t     biPlanes;          // bitplanes
    int16_t     biBitCount;        // resolution 
    int32_t     biCompression;     // compression
    int32_t     biSizeImage;       // size of the image
    int         biXPelsPerMeter;   // pixels per meter X
    int         biYPelsPerMeter;   // pixels per meter Y
    int32_t     biClrUsed;         // colors used
    int32_t     biClrImportant;    // important colors

} BITMAPINFOHEADER;

class Bitmap
{

public:
    Bitmap();
    Bitmap(const uint width, 
                const uint height, 
                const uint padded_size);
    ~Bitmap();
    void open( const char* path);
    void save( const char* path);
    uint get_width();
    uint get_height();
    uint get_padded_size();

    const BITMAPFILEHEADER* getFileHeader();
    const BITMAPINFOHEADER* getInfoHeader();
    uint8_t * getRawData();


private:

    void paddedToRGB(const uint8_t * source,
                     uint8_t* target);

    void RGBtoPadded(const uint8_t * target,
                     uint8_t* source);
    void generate_headers();


private:
	uint m_width;
	uint m_height;
  uint m_padded_size;
  uint m_size;

  BITMAPFILEHEADER m_bitmap_file_header;
  BITMAPINFOHEADER m_bitmap_info_header;

  //TODO: SHOULD I USE SMART POINTERS?
  uint8_t * m_padded_buffer_data;
  uint8_t * m_buffer_data;  
};

#endif