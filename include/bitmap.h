/** 
@file bitmap.h
@brief Header for the Bitmap class.

This header contains the declarartion of the Bitmap class,
this class is not intended in being a production ready bitmpa class 
or anything, If I wanted that I would have used a library, the reason
why this class exists is first a learning purpose in reading a binary file 
and second to have a basic class to read and save bitmap.
The funcionalities are limited in the barebone essential I needed for doing
my parallel stuff.
I took some code and a lot of help from this website:
http://tipsandtricks.runicsoft.com/Cpp/BitmapTutorial.html

@author Marco Giordano
@bug Sometimes when the image width-height is not even , or some weird size the reader, freask out and reads the image wrong,
	 like it grabs a piece of the left image , like a 1/4 ~1/6 of it  and place it on the right, like if you wrap-slided the image,
	 not going to fix that for now, I am not interested on getting a bad ass bmp class up and running
 */

#include <string>
#include <cstdint>
#if defined(WIN32)
//if on windows include the windwos.h so we have the declaration of many of the datatype needed
#include <Windows.h>
#endif



using namespace std;

#ifndef __PARALLEL_IMAGE_BITMAP__
#define __PARALLEL_IMAGE_BITMAP__



#if defined(LINUX)
/**
@brief Structure defining the header of the bitmap file

The __attribute__((__packed__))  is needed for gcc to not 
add padding to the structure which would make the binary read
offsetted
*/
typedef struct   BITMAPFILEHEADER
{
  int16_t  bfType;          /** type of the BMP must be BM*/
  int32_t bfSize;           /** size of the whole .bmp file */
  int16_t  bfReserved1;     /** reserved, must be 0*/
  int16_t  bfReserved2;     /** reserved, must be 0*/
  int32_t bfOffBits;        /** size of the structure*/
} __attribute__((__packed__)) BITMAPFILEHEADER;
#endif

#if defined(LINUX)
/**
@brief Structure defining the header of the image content

The __attribute__((__packed__))  is needed for gcc to not 
add padding to the structure which would make the binary read
offsetted
*/

typedef struct BITMAPINFOHEADER
{
    int         biSize;            /** size of the structure*/
    int         biWidth;           /** image width*/
    int         biHeight;          /** image height*/
    int16_t     biPlanes;          /** bitplanes*/
    int16_t     biBitCount;        /** resolution */
    int32_t     biCompression;     /** compression*/
    int32_t     biSizeImage;       /** size of the image*/
    int         biXPelsPerMeter;   /** pixels per meter X*/
    int         biYPelsPerMeter;   /** pixels per meter Y*/
    int32_t     biClrUsed;         /** colors used*/
    int32_t     biClrImportant;    /** important colors*/

} __attribute__((__packed__))  BITMAPINFOHEADER;

#endif

/** 
@brief Class for manipulating bitmaps.

This is the class in charge of opening and loading the BMP,
once it does that it provides us with a nice buffer perfect for 
being mainpulate both on cpu and gpu once we are done we can just save
the bmp to check the result

 */
class Bitmap
{

public:
  /** 
  @brief The default constructor
   */ 
  Bitmap();
  /** 
  @brief Alternative constructor that generate 
  for us a dummy buffer ready to be used
  @param width: the width in pixel of the image
  @param heigh: the height in pixel of the image
  @param padded_size: the size of the padded buffer, taking into account
                      padded scanlines 
   */ 
  Bitmap(const unsigned int width, 
              const unsigned int height, 
              const unsigned int padded_size);
  /** 
  @brief The destructor
   */ 
  ~Bitmap();

  /** 
  @brief Open an image from file
  @param path: the path to the file, if not valid exeption is raised
  @throw std::runtime_error , if file not valid
   */ 
  void open( const char* path);

  /** 
  @brief Save an image from buffer
  @param path: the path to the file to be saved, if not valid exeption is raised
  @throw std::runtime_error , if file not valid
   */ 
  void save( const char* path);

  /** 
  @brief Get the width of the loaded/initialized image
  @return unsigned int
   */ 
  unsigned int get_width();
  
  /** 
  @brief Get the height of the loaded/initialized image
  @return unsigned int
   */ 
  unsigned int get_height();

  /** 
  @brief Get the the padded_size of the buffer of the opened image
  @return unsigned int
   */ 
  unsigned int get_padded_size();

  /** 
  @brief Get a pointer to the loaded/initialized header
  @return BITMAPFILEHEADER*
   */ 
  const BITMAPFILEHEADER* getFileHeader();

  /** 
  @brief Get a pointer to the loaded/initialized header
  @return BITMAPINFOHEADER*
   */ 
  const BITMAPINFOHEADER* getInfoHeader();

  /** 
  @brief Get a pointer to bmp not padded buffer
  @return uint8_t*
   */ 
  uint8_t* getRawData();


private:
  /** 
  @brief private copy constructor
  For now copying with copy constructor is not supported thus made
  privet
   */   
  Bitmap(Bitmap& bmp);

  /** 
  @brief convert a padded buffer to an flat rgb buffer
  @param source: pointer to the source padded buffer
  @param target: pointer to the target not padded buffer
   */ 
  void paddedToRGB(const uint8_t * source,
                   uint8_t* target);

  /** 
  @brief convert a padded buffer to an flat rgb buffer
  @param target: pointer to the target padded buffer
  @param source: pointer to the source not padded buffer
   */ 
  void RGBtoPadded(const uint8_t * target,
                   uint8_t* source);

  /** 
  @brief Generates default headers for the BMP image
   */ 
  void generate_headers();


private:
  //internal width of the image
	unsigned int m_width;
  //internal height of the image
	unsigned int m_height;
  //internal padded size of the image
  unsigned int m_padded_size;
  //internal size of the not padded buffer
  unsigned int m_size;

  //internal file header
  BITMAPFILEHEADER m_bitmap_file_header;
  //internal info header
  BITMAPINFOHEADER m_bitmap_info_header;

  //TODO: SHOULD I USE SMART POINTERS?
  //internal pointer to the padded buffer, unprocessed so upside, down
  uint8_t * m_padded_buffer_data;
  //internal pointer to the not padded buffer, the buffer goes left to right, top to bottom
  uint8_t * m_buffer_data;  
};

#endif