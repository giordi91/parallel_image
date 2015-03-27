#include <cstdint> // uint8_t declaration
#include <gmock/gmock.h>

#include <core/filter_manager.h>
#include <core/filter.h>
#include <core/bitmap.h>
using namespace testing;
//generating mock classes for the filters

class Foo
{
	int test_mock(int value){return 1;};

};


class FilterMockA : public Filter 
{
  MOCK_METHOD2(compute_serial, void(const uint8_t * source,
  										  uint8_t* target));
  MOCK_METHOD2(compute_tbb, void(const uint8_t * source,
  	                                   uint8_t* target));
  MOCK_METHOD2(compute_cuda, void(const uint8_t * source,
  								        uint8_t* target));
};

class FilterMockB : public Filter 
{
  MOCK_METHOD2(compute_serial, void(const uint8_t * source,
  										  uint8_t* target));
  MOCK_METHOD2(compute_tbb, void(const uint8_t * source,
  	                                   uint8_t* target));
  MOCK_METHOD2(compute_cuda, void(const uint8_t * source,
  								        uint8_t* target));
};

class MockBmp: public Bitmap
{


};

TEST(filter_manager, constructor)
{
	Bitmap * bmp= new Bitmap() ;
	Filter_manager fm(bmp);
}

TEST(filter_manager, add_filter)
{
	Bitmap * bmp= new Bitmap() ;
	Filter_manager fm(bmp);

}

//able to allocate and manage filters
//remove filters by id, other get shifted
//able to manage cache trough external classes
//albe to manage computation type cpu/cppu/gpu