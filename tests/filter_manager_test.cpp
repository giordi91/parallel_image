#include <cstdint> // uint8_t declaration
#include <gmock/gmock.h>
#include <iostream>
#include <stdexcept>      // std::invalid_argument
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

struct Filter_basic_fixture_Test : public Test 
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

struct Filter_built_fixture_Test : public Test 
{
	//allocating object on the heap, now no need to delete those 
	//in the constructor since the ownership will pass to the manager,
	//and it will take care to clean up after itself, both for the bmp
	//image and the filters
	MockBmp * bmp;
	FilterMockA * fil1;
	FilterMockB * fil2;
	FilterMockB * fil3;
	FilterMockA * fil4;

	Filter_manager  * fm;

	virtual void SetUp()
    { 
    	bmp= new MockBmp ;
    	fm = new Filter_manager(bmp);

    	fil1 =new FilterMockA;
    	fil2 =new FilterMockB;
    	fil3 =new FilterMockB;
    	fil4 =new FilterMockA;

		fm->add_filter(fil1);
		fm->add_filter(fil2);
		fm->add_filter(fil3);
		fm->add_filter(fil4);
    }

    virtual void TearDown()
	{
		delete fm;
	}

};


TEST_F(Filter_basic_fixture_Test, constructor)
{
	Filter_manager fm=Filter_manager(bmp);
}

TEST_F(Filter_basic_fixture_Test, add_filters_and_check_size)
{

	Filter_manager fm=Filter_manager(bmp);
	EXPECT_EQ(fm.stack_size(),0);
	fm.add_filter(fil1);
	EXPECT_EQ(fm.stack_size(),1);
	fm.add_filter(fil2);
	fm.add_filter(fil3);
	EXPECT_EQ(fm.stack_size(),3);

}

TEST_F(Filter_basic_fixture_Test, subscription_operator)
{

	Filter_manager fm=Filter_manager(bmp);
	fm.add_filter(fil1);
	fm.add_filter(fil2);
	fm.add_filter(fil3);
	EXPECT_EQ(fm[0] ,fil1);
	EXPECT_EQ(fm[2] ,fil3);
}

TEST_F(Filter_built_fixture_Test, remove_filter_from_stack)
{
	fm->remove_filter(0);
	EXPECT_EQ(fm->stack_size(),3);
	fm->remove_filter(0);
	fm->remove_filter(1);
	EXPECT_EQ(fm->stack_size(),1);
	fm->remove_filter(0);
	EXPECT_EQ(fm->stack_size(),0);
	EXPECT_THROW(fm->remove_filter(0), std::invalid_argument);
}

TEST_F(Filter_built_fixture_Test, pop_filter_from_stack)
{
	Filter * temp1,* temp2, * temp3,* temp4;
	temp3 = fm->pop_filter(2);
	EXPECT_EQ(temp3, fil3);
	EXPECT_EQ(fm->stack_size(),3);

	temp1 = fm->pop_filter(0);
	EXPECT_EQ(temp1, fil1);
	EXPECT_EQ(fm->stack_size(),2);	

	temp4 = fm->pop_filter(1);
	EXPECT_EQ(temp4, fil4);
	EXPECT_EQ(fm->stack_size(),1);	

	temp2 = fm->pop_filter(0);
	EXPECT_EQ(temp4, fil4);
	EXPECT_EQ(fm->stack_size(),0);
	
	EXPECT_THROW(fm->pop_filter(0), std::invalid_argument);

	//deleteing manually the filters since the ownership is not on
	//the manager anymore
	delete temp3;
	delete temp1;
	delete temp4;
	delete temp2;


}



//able to allocate and manage filters X
//access with subscription operator X
//remove filters by id, other get shifted X
//able to manage cache trough external classes
//albe to manage computation type cpu/cppu/gpu