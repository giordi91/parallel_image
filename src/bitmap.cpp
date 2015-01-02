#include <bitmap.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>



using namespace std;

Bitmap::Bitmap():m_width(0), m_height(0), m_padded_size(0), m_size(0),
                m_padded_buffer_data(NULL),
                m_buffer_data(NULL)
{

}

Bitmap::Bitmap( const unsigned int width, 
                const unsigned int height, 
                const unsigned int padded_size):
                m_width(width),m_height(height), 
                m_padded_size(padded_size), 
                m_size(width*height*3),
                m_padded_buffer_data(NULL),
                m_buffer_data(NULL)
{
    //generate a default header
    generate_headers();

    //allocating the buffers
    m_padded_buffer_data = new uint8_t[m_padded_size];
    m_buffer_data = new uint8_t[m_size];

}


Bitmap::~Bitmap()
{
    //checking if the pointer have been used
    //if so free the data    
    if (m_padded_buffer_data)
    {
        delete [] m_padded_buffer_data;
    }
    if (m_buffer_data)
    {
        delete [] m_buffer_data;
    }
}   

void Bitmap::open(const char* path)
{
    ifstream f(path,ifstream::binary | ifstream::in);

    //checking if the path is valid , if not raise an exception
    if(!f.good())
    {
        throw std::runtime_error("The input file path is not valid");
        return;
    }


    //read in the header file header    
    f.read((char*)&m_bitmap_file_header, sizeof(BITMAPFILEHEADER));

    //read in the image header
    f.read((char*)&m_bitmap_info_header, sizeof(m_bitmap_info_header));

    //setting pointer to beginning of the pixel data
    f.seekg(m_bitmap_file_header.bfOffBits, ios::beg);

    //compute the size of the pixel data
    m_padded_size = m_bitmap_file_header.bfSize -  
                                m_bitmap_file_header.bfOffBits;

    //allocating padded buffer data
    m_padded_buffer_data = new uint8_t[m_padded_size];
    //reading padded data
    f.read((char*)m_padded_buffer_data,m_padded_size);
    f.close();

    //intializing internal data to work with the functions
    m_width = m_bitmap_info_header.biWidth;
    m_height = m_bitmap_info_header.biHeight;
    m_size = m_width*m_height*3;
    //allocating buffer data
    m_buffer_data = new uint8_t[m_size];

    //converting the padded and reversed data to basic rgb data
    paddedToRGB(m_padded_buffer_data, m_buffer_data);

}

void Bitmap::save(const char* path)
{
    //creating the needed stream
    ofstream f(path,ofstream::binary | ofstream::out);

    if(!f.good())
    {
        throw std::runtime_error("Impossible to save in the given path");
        return;
    }

    //writing the headers informations of the bmp
    f.write((char*)&m_bitmap_file_header, sizeof(BITMAPFILEHEADER));
    f.write((char*)&m_bitmap_info_header, sizeof(BITMAPINFOHEADER));
    
    //computing the padding of the scanline
    int padding = 0;
    int scanlinebytes = m_width * 3;
    while ( ( scanlinebytes + padding ) % 4 != 0 ) 
        ++padding;
    int psw = scanlinebytes + padding;

    //converting rgb data into padded data ready to be saved
    paddedToRGB(m_buffer_data, m_padded_buffer_data);

    //go to the offset of the BMP
    f.seekp(m_bitmap_file_header.bfOffBits, ofstream::beg);
    //write the data
    f.write((char*)m_padded_buffer_data, m_height*psw);
    //close the file
    f.close();

}

unsigned int Bitmap::get_width()
{
    return m_width;
}

unsigned int Bitmap::get_height()
{
    return m_height;
}

unsigned int Bitmap::get_padded_size()
{
    return m_padded_size;
}

const BITMAPFILEHEADER* Bitmap::getFileHeader()
{
    return &m_bitmap_file_header;
}

const BITMAPINFOHEADER* Bitmap::getInfoHeader()
{
    return &m_bitmap_info_header;
}

void Bitmap::paddedToRGB(const uint8_t * source,
                     uint8_t* target)
{
        //computing padding
    int padding = 0;
    int scanlinebytes = m_width * 3;
    while ( ( scanlinebytes + padding ) % 4 != 0 )
        ++padding;

    int psw = scanlinebytes + padding;

    //converting raw buffer data to a rgb array
    long bufpos = 0;   
    long newpos = 0;
    for ( unsigned int y = 0; y < m_height; ++y )
        for ( unsigned int x = 0; x < 3 * m_width; x+=3 )
        {
            newpos = y * 3 * m_width + x;     
            bufpos = ( m_height - y - 1 ) * psw + x;

            target[newpos] = source[bufpos + 2];       
            target[newpos + 1] = source[bufpos+1]; 
            target[newpos + 2] = source[bufpos];               
        }
}

void Bitmap::RGBtoPadded(const uint8_t * source,
                     uint8_t* target)
{
    int padding = 0;
    int scanlinebytes = m_width * 3;
    while ( ( scanlinebytes + padding ) % 4 != 0 ) 
        ++padding;
    int psw = scanlinebytes + padding;

    long bufpos = 0;   
    long newpos = 0;
    for ( unsigned int y = 0; y < m_height; ++y )
        for ( unsigned int x = 0; x < 3 * m_width; x+=3 )
        {
            bufpos = y * 3 * m_width + x;     // position in original buffer
            newpos = ( m_height - y - 1 ) * psw + x; // position in padded buffer
            target[newpos] = source[bufpos+2];       // swap r and b
            target[newpos + 1] = source[bufpos + 1]; // g stays
            target[newpos + 2] = source[bufpos];     // swap b and r
        }
}


uint8_t* Bitmap::getRawData()
{
    return m_buffer_data;

}

void Bitmap::generate_headers()
{
    m_bitmap_file_header = BITMAPFILEHEADER();
    m_bitmap_info_header = BITMAPINFOHEADER();
    //init with zeros
    memset ( &m_bitmap_file_header, 0, sizeof (BITMAPFILEHEADER));
    memset ( &m_bitmap_info_header, 0, sizeof (BITMAPINFOHEADER));

    //write default buffer info

    //file header
    m_bitmap_file_header.bfType = 0x4d42;       // 0x4d42 = 'BM'
    m_bitmap_file_header.bfReserved1 = 0;
    m_bitmap_file_header.bfReserved2 = 0;
    m_bitmap_file_header.bfSize = int32_t(sizeof(BITMAPFILEHEADER)) + 
        int32_t(sizeof(BITMAPINFOHEADER)) + m_padded_size;
    m_bitmap_file_header.bfOffBits = 0x36;

    //info header
    m_bitmap_info_header.biSize = sizeof(BITMAPINFOHEADER);
    m_bitmap_info_header.biWidth = m_width;
    m_bitmap_info_header.biHeight = m_height;
    m_bitmap_info_header.biPlanes = 1;  
    m_bitmap_info_header.biBitCount = 24;
    //value of ‘BI_RGB’ means not compresssed
    m_bitmap_info_header.biCompression = 0;    
    m_bitmap_info_header.biSizeImage = 0;
    m_bitmap_info_header.biXPelsPerMeter = 0x0ec4;  
    m_bitmap_info_header.biYPelsPerMeter = 0x0ec4;     
    m_bitmap_info_header.biClrUsed = 0; 
    m_bitmap_info_header.biClrImportant = 0; 

}