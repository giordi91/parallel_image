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
    delete [] m_padded_buffer_data;
    delete [] m_buffer_data;
    



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
    std::cout<<"type :"<<(m_bitmap_file_header.bfType ==  0x4D42)<<endl;
    std::cout<<"info header size"<< m_bitmap_file_header.bfSize<<endl;
    std::cout<<"offset :"<<m_bitmap_file_header.bfOffBits<<endl;

    //.seekg(m_bitmap_file_header.bfOffBits, ios::beg);

    f.read((char*)&m_bitmap_info_header, sizeof(m_bitmap_info_header));
    std::cout<<m_bitmap_info_header.biHeight<<endl;

    f.seekg(m_bitmap_file_header.bfOffBits, ios::beg);
    int buff_size = m_bitmap_file_header.bfSize -  
                                m_bitmap_file_header.bfOffBits;
    cout <<m_bitmap_info_header.biSizeImage<<endl;

    m_padded_buffer_data = new uint8_t[buff_size];
    

    f.read((char*)m_padded_buffer_data, m_bitmap_info_header.biSizeImage);

    m_width = m_bitmap_info_header.biWidth;
    m_height = m_bitmap_info_header.biHeight;

    m_buffer_data = new uint8_t[m_width*m_height*3];

    int padding = 0;
    int scanlinebytes = m_width * 3;
    while ( ( scanlinebytes + padding ) % 4 != 0 )
        padding++;

    int psw = scanlinebytes + padding;

    std::cout<<padding<<endl;

    long bufpos = 0;   
    long newpos = 0;
    for ( int y = 0; y < m_height; y++ )
        for ( int x = 0; x < 3 * m_width; x+=3 )
        {
            newpos = y * 3 * m_width + x;     
            bufpos = ( m_height - y - 1 ) * psw + x;

            m_buffer_data[newpos] = m_padded_buffer_data[bufpos + 2];       
            m_buffer_data[newpos + 1] = m_padded_buffer_data[bufpos+1]; 
            m_buffer_data[newpos + 2] = m_padded_buffer_data[bufpos];     
        }

    // std::cout<<int(m_buffer_data[0])<<" "<<int(m_buffer_data[1])<<" "<<int(m_buffer_data[2])<<" "<<endl;
    // std::cout<<int(m_buffer_data[3])<<" "<<int(m_buffer_data[4])<<" "<<int(m_buffer_data[5])<<" "<<endl;
    // std::cout<<int(m_buffer_data[6])<<" "<<int(m_buffer_data[7])<<" "<<int(m_buffer_data[8])<<" "<<endl;
    // std::cout<<int(m_buffer_data[9])<<" "<<int(m_buffer_data[10])<<" "<<int(m_buffer_data[11])<<" "<<endl;
    
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
// 
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