#include <iostream>

using namespace std;

int main( int argc, char* argv[])
{
	cout << "There are " << argc << " arguments:" << endl;
 
    // Loop through each argument and print its number and value
    for (int nArg=0; nArg < argc; nArg++)
        cout << nArg << " " << argv[nArg] << endl;
 
    return 0;

}