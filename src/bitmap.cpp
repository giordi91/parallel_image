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



    //first step read two chars
    int infoHeaderSize;
    char temp[2];

    f.read((char*)&temp, sizeof(char)*2);
    f.read((char*)&m_byte_size, sizeof(int));
    m_bmp_type = temp;

    //skip 8 bytes of junk
    f.seekg(4, ios::cur);

    int offset;

    f.read((char*)&offset, sizeof(int));
    f.read((char*)&infoHeaderSize, sizeof(int));
    f.read((char*)&m_width, sizeof(int));
    f.read((char*)&m_height, sizeof(int));
    // f.seekg(infoHeaderSize- sizeof(int)*3 + 24, ios::cur);

    f.seekg(offset , ios::beg);
    uint8_t r,g,b;
    f.read((char*)&r, sizeof(uint8_t));
    f.read((char*)&g, sizeof(uint8_t));
    f.read((char*)&b, sizeof(uint8_t));


    std::cout<<m_bmp_type<<endl;
    std::cout<<m_byte_size<<endl;
    std::cout<<"info header size"<< infoHeaderSize<<endl;
    std::cout<<m_width<<endl;
    std::cout<<m_height<<endl;
    std::cout<<int(r)<<endl;
    std::cout<<int(g)<<endl;
    std::cout<<int(b)<<endl;
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
