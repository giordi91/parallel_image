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
    char bmpType[2];
    int bmpSize;
    int infoHeaderSize;
    int bmpWidth;
    int bmpHeight;

    f.read((char*)&bmpType, sizeof(char)*2);
    f.read((char*)&bmpSize, sizeof(int));

    //skip 8 bytes of junk
    f.seekg(8, ios::cur);
    f.read((char*)&infoHeaderSize, sizeof(int));
    f.read((char*)&bmpWidth, sizeof(int));
    f.read((char*)&bmpHeight, sizeof(int));


    std::cout<<bmpType<<endl;
    std::cout<<bmpSize<<endl;
    std::cout<<infoHeaderSize<<endl;
    std::cout<<bmpWidth<<endl;
    std::cout<<bmpHeight<<endl;
    // f


}