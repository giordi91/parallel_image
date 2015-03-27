#include <cstdint> // uint8_t declaration
#include <gmock/gmock.h>
#include <iostream>
#include <core/filter_manager.h>
#include <core/filter.h>
#include <core/bitmap.h>
using namespace testing;


//generating mock classes for the filters
class FilterMockA : public Filter 
{
  public:

  MOCK_METHOD2(compute_serial, void(const uint8_t * source,
  										  uint8_t* target));
  MOCK_METHOD2(compute_tbb, void(const uint8_t * source,
  	                                   uint8_t* target));
  MOCK_METHOD2(compute_cuda, void(const uint8_t * source,
  								        uint8_t* target));
};

class FilterMockB : public Filter 
{
  public:

  MOCK_METHOD2(compute_serial, void(const uint8_t * source,
  										  uint8_t* target));
  MOCK_METHOD2(compute_tbb, void(const uint8_t * source,
  	                                   uint8_t* target));
  MOCK_METHOD2(compute_cuda, void(const uint8_t * source,
  								        uint8_t* target));
};


class MockBmp: public Bitmap
{
	public:
	~MockBmp(){std::cout<<"YAAGH"<<std::endl;}
};

struct Filter_manager_Fixture_Test : public Test 
{
	//allocating object on the heap, now no need to delete those 
	//in the constructor since the ownership will pass to the manager,
	//and it will take care to clean up after itself, both for the bmp
	//image and the filters

	MockBmp * bmp= new MockBmp ;
	FilterMockA * fil1 =new FilterMockA;
	FilterMockA * fil2=new FilterMockA;
	FilterMockB * fil3=new FilterMockB;
	FilterMockB * fil4=new FilterMockB;


};

TEST_F(Filter_manager_Fixture_Test, constructor)
{
	Filter_manager fm(bmp);
}

TEST_F(Filter_manager_Fixture_Test, add_filters_and_check_size)
{

	Filter_manager fm(bmp);
	EXPECT_EQ(fm.stack_size(),0);
	fm.add_filter(fil1);
	EXPECT_EQ(fm.stack_size(),1);
	fm.add_filter(fil2);
	fm.add_filter(fil3);
	EXPECT_EQ(fm.stack_size(),3);

}

//able to allocate and manage filters
//access with subscription operator
//remove filters by id, other get shifted
//able to manage cache trough external classes
//albe to manage computation type cpu/cppu/gpu