#include <bitmap.h>
#include <iostream>
#include <fstream>
#include <stdexcept>


using namespace std;

//SHOULD I USE SMART POINTERS?
Bitmap::Bitmap():m_padded_buffer_data(NULL),
                m_buffer_data(NULL)
{

}

Bitmap::~Bitmap()
{
    //checking if the pointer have been used
    //if so free the data    
    if (!m_padded_buffer_data == NULL)
    {
        delete [] m_padded_buffer_data;
    }
    if (!m_buffer_data == NULL)
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
    int buff_size = m_bitmap_file_header.bfSize -  
                                m_bitmap_file_header.bfOffBits;

    //allocating padded buffer data
    m_padded_buffer_data = new uint8_t[buff_size];
    //reading padded data
    f.read((char*)m_padded_buffer_data,buff_size);
    f.close();

    //intializing internal data to work with the functions
    m_width = m_bitmap_info_header.biWidth;
    m_height = m_bitmap_info_header.biHeight;

    //allocating buffer data
    m_buffer_data = new uint8_t[m_width*m_height*3];

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
        padding++;
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

int Bitmap::width()
{
    return m_width;
}

int Bitmap::height()
{
    return m_height;
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
        padding++;

    int psw = scanlinebytes + padding;

    //converting raw buffer data to a rgb array
    long bufpos = 0;   
    long newpos = 0;
    for ( int y = 0; y < m_height; y++ )
        for ( int x = 0; x < 3 * m_width; x+=3 )
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
        padding++;
    int psw = scanlinebytes + padding;

    long bufpos = 0;   
    long newpos = 0;
    for ( int y = 0; y < m_height; y++ )
        for ( int x = 0; x < 3 * m_width; x+=3 )
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

void Bitmap::setRawData(uint8_t * buffer)
{
    m_buffer_data= buffer;
}