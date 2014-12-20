#include <bitmap.h>
#include <iostream>
#include <fstream>


using namespace std;

Bitmap::Bitmap()
{
    cout<<"constructor bitches"<<endl;

}


Bitmap::~Bitmap()
{

    cout <<"destructor bitchessssss"<<endl;

}

void Bitmap::open(const char* path)
{
    ifstream f(path,ifstream::binary | ifstream::in);

    cout<<"opening the file..."<<endl;
    if(!f.good())
    {
        cout<<"path not valid"<<endl;
        return;
    }
    else
    {
        cout<<"file is valid! woot"<<endl;
    }

    
    f.read((char*)&m_bitmap_file_header, sizeof(BITMAPFILEHEADER));
    std::cout<<"type :"<<m_bitmap_file_header.bfType<<endl;
    std::cout<<"info header size"<< m_bitmap_file_header.bfSize<<endl;
    std::cout<<"offset :"<<m_bitmap_file_header.bfOffBits<<endl;

    //.seekg(m_bitmap_file_header.bfOffBits, ios::beg);

    f.read((char*)&m_bitmap_info_header, sizeof(m_bitmap_info_header));
    std::cout<<m_bitmap_info_header.biHeight<<endl;

    //first step read two chars
    // int infoHeaderSize;
    // char temp[2];

    // f.read((char*)&temp, sizeof(char)*2);
    // f.read((char*)&m_byte_size, sizeof(int));
    // m_bmp_type = temp;

    // //skip 8 bytes of junk
    // f.seekg(4, ios::cur);

    // int offset;

    // f.read((char*)&offset, sizeof(int));
    // f.read((char*)&infoHeaderSize, sizeof(int));
    // f.read((char*)&m_width, sizeof(int));
    // f.read((char*)&m_height, sizeof(int));
    // // f.seekg(infoHeaderSize- sizeof(int)*3 + 24, ios::cur);

    // f.seekg(offset , ios::beg);
    // uint8_t r,g,b;
    // f.read((char*)&r, sizeof(uint8_t));
    // f.read((char*)&g, sizeof(uint8_t));
    // f.read((char*)&b, sizeof(uint8_t));


    // std::cout<<"type :"<<m_bmp_type<<endl;
    // std::cout<<"info header size"<< infoHeaderSize<<endl;
    // std::cout<<"offset :"<<offset<<endl;
    // std::cout<<m_width<<endl;
    // std::cout<<m_height<<endl;
    // std::cout<<int(r)<<endl;
    // std::cout<<int(g)<<endl;
    // std::cout<<int(b)<<endl;
    // std::cout<<g<<endl;
    // std::cout<<b<<endl;
    // std::cout<<offset<<endl;
    // f


    
}

int Bitmap::width()
{
    return m_width;
}

int Bitmap::height()
{
    return m_height;
}

// char Bitmap::type()
// {
//     return m_bmp_type;
// }

int Bitmap::size()
{
    return m_byte_size;
}

const BITMAPFILEHEADER* Bitmap::getFileHeader()
{
    return &m_bitmap_file_header;

}

const BITMAPINFOHEADER* Bitmap::getInfoHeader()
{
    return &m_bitmap_info_header;

}