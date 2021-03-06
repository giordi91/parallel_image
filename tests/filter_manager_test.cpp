#include <cstdint> // uint8_t declaration
#include <gmock/gmock.h>
#include <iostream>
#include <stdexcept>      // std::invalid_argument
#include <core/filter_manager.h>
#include <core/filter.h>
#include <core/bitmap.h>
#include <filters/bw_filter.h>
#include <filters/gaussian_filter.h>
#include <filters/sharpen_filter.h>
#include <filters/edge_detection_filter.h>
using namespace testing;


//generating mock classes for the filters
class FilterMockA : public Filter 
{
  public:

  MOCK_METHOD2(compute_serial, void(const uint8_t * source,
  										  uint8_t* target));
  MOCK_METHOD2(compute_tbb, void(const uint8_t * source,
  	                                   uint8_t* target));
  MOCK_METHOD2(compute_cuda, void( uint8_t * source,
  								        uint8_t* target));
  MOCK_METHOD0(update_data, void());
  MOCK_METHOD0(get_type, std::string ());

};

class FilterMockB : public Filter 
{
  public:

  MOCK_METHOD2(compute_serial, void(const uint8_t * source,
  										  uint8_t* target));
  MOCK_METHOD2(compute_tbb, void(const uint8_t * source,
  	                                   uint8_t* target));
  MOCK_METHOD2(compute_cuda, void( uint8_t * source,
  								        uint8_t* target));
  MOCK_METHOD0(update_data, void ());

  MOCK_METHOD0(get_type, std::string ());
  

};


class MockBmp: public Bitmap
{};

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

    	fil1 =new FilterMockA();
    	fil2 =new FilterMockB();
    	fil3 =new FilterMockB();
    	fil4 =new FilterMockA();

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

struct Heavy_Filter_computation_test_fixture: public Test 
{

	Bitmap * testbmp;
	Bw_filter * bw ;
	Gaussian_filter * gf ;
	Sharpen_filter * sp ;
	Edge_detection_filter * ed ;
	Filter_manager * fm;
	virtual void SetUp()
    {
		testbmp = new Bitmap;
	    try
	    {
	        //E:/WORK_IN_PROGRESS/C/parallel_image/data
	        ///user_data/WORK_IN_PROGRESS/parallel_image/data/jessy.bmp
	        ///home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessy.bmp
	        testbmp->open("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/test_01.bmp");
	    }
	    catch(std::runtime_error &e)
	    {
	        std::cout<<e.what()<<endl;
	        #if defined(WIN32)
	        system ("PAUSE");
	        #endif
	        return;
	    } 

		//gather the data
	    unsigned int width = testbmp->get_width();
	    unsigned int height = testbmp->get_height();

	    bw = new Bw_filter ( width, height);
	    gf = new Gaussian_filter(width,height,2.0f);
	    sp= new Sharpen_filter(width,height);
	    ed = new Edge_detection_filter(width,height,2);

		fm = new Filter_manager(testbmp);
	    fm->add_filter(bw);
	    fm->add_filter(ed);
	    fm->add_filter(sp);
	    fm->add_filter(gf);
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

TEST_F(Filter_built_fixture_Test, set_get_computation)
{
	EXPECT_EQ(fm->get_compute_type(), Filter_manager::SERIAL);

	fm->set_compute_type(Filter_manager::TBB);
	EXPECT_EQ(fm->get_compute_type(), Filter_manager::TBB);

	fm->set_compute_type(Filter_manager::CUDA);
	EXPECT_EQ(fm->get_compute_type(), Filter_manager::CUDA);
}

TEST_F(Filter_built_fixture_Test, basic_evaluate_stack)
{
	//weird behavior??, why if i only set an expected call 
	//from fil1 i get 3 extra unexpected call? if i add aexepcted
	//calls for all the other filters I am good

	EXPECT_CALL(*fil1, compute_serial(_,_)).Times(1);
	EXPECT_CALL(*fil2, compute_serial(_,_)).Times(1);
	EXPECT_CALL(*fil3, compute_serial(_,_)).Times(1);
	EXPECT_CALL(*fil4, compute_serial(_,_)).Times(1);
	fm->evaluate_stack();  
}

TEST_F(Filter_built_fixture_Test, evaluate_stack_and_changed_comp_type)
{


	EXPECT_CALL(*fil1, compute_serial(_,_)).Times(1);
	EXPECT_CALL(*fil2, compute_serial(_,_)).Times(1);
	EXPECT_CALL(*fil3, compute_serial(_,_)).Times(1);
	EXPECT_CALL(*fil4, compute_serial(_,_)).Times(1);

	EXPECT_CALL(*fil1, compute_tbb(_,_)).Times(2);
	EXPECT_CALL(*fil2, compute_tbb(_,_)).Times(2);
	EXPECT_CALL(*fil3, compute_tbb(_,_)).Times(2);
	EXPECT_CALL(*fil4, compute_tbb(_,_)).Times(2);

	EXPECT_CALL(*fil1, compute_cuda(_,_)).Times(1);
	EXPECT_CALL(*fil2, compute_cuda(_,_)).Times(1);
	EXPECT_CALL(*fil3, compute_cuda(_,_)).Times(1);
	EXPECT_CALL(*fil4, compute_cuda(_,_)).Times(1);
	
	fm->set_compute_type(Filter_manager::TBB);
	fm->evaluate_stack();  
	fm->set_compute_type(Filter_manager::SERIAL);
	fm->evaluate_stack();  
	fm->set_compute_type(Filter_manager::CUDA);
	fm->evaluate_stack();  
	fm->set_compute_type(Filter_manager::TBB);
	fm->evaluate_stack();  
}

TEST_F(Filter_built_fixture_Test, add_filter_using_name)
{
	fm->add_filter_by_name("Edge_detection_filter");
	EXPECT_EQ(fm->stack_size(),5);
	fm->add_filter_by_name("Bw_filter");
	EXPECT_EQ(fm->stack_size(),6);
	fm->add_filter_by_name("Sharpen_filter");
	EXPECT_EQ(fm->stack_size(),7);
	fm->add_filter_by_name("Bw_filter");
	EXPECT_EQ(fm->stack_size(),8);
	fm->add_filter_by_name("Bw_filter");
	EXPECT_EQ(fm->stack_size(),9);
}

TEST_F(Filter_built_fixture_Test, add_filter_using_name_failure)
{
	
	EXPECT_THROW(fm->add_filter_by_name("VOID_FILTER"),
										std::invalid_argument);
	EXPECT_EQ(fm->stack_size(),4);
	fm->add_filter_by_name("Edge_detection_filter");
	EXPECT_EQ(fm->stack_size(),5);
	EXPECT_THROW(fm->add_filter_by_name("VOID_FILTER"),
										std::invalid_argument);
	EXPECT_EQ(fm->stack_size(),5);
	EXPECT_THROW(fm->add_filter_by_name("VOID_FILTER"),
										std::invalid_argument);
	EXPECT_EQ(fm->stack_size(),5);
}



TEST_F(Heavy_Filter_computation_test_fixture, testing_serial_filters)
{	
    fm->evaluate_stack();
    fm->save_stack_output("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/OUT_testing_serial_filters.bmp");
}


TEST_F(Heavy_Filter_computation_test_fixture, testing_TBB_filters)
{	
    fm->set_compute_type(Filter_manager::TBB);
    fm->evaluate_stack();
    fm->save_stack_output("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/OUT_testing_TBB_filters.bmp");
}

TEST_F(Heavy_Filter_computation_test_fixture, testing_CUDA_filters)
{	
    fm->set_compute_type(Filter_manager::CUDA);
    fm->evaluate_stack();
    fm->save_stack_output("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/OUT_testing_CUDA_filters.bmp");
}


//able to allocate and manage filters X
//access with subscription operator X
//remove filters by id, other get shifted X
//set,query computation type X
//able to manage cache trough external classes
//albe to manage computation type cpu/cppu/gpu X
//add filters by name (factory behavior)
