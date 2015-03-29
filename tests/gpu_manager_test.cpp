#include <cstdint> // uint8_t declaration
#include <gmock/gmock.h>
#include <iostream>
#include <stdexcept>      // std::invalid_argument

#include <core/bitmap.h>
#include <core/GPU_manager.h>
using namespace testing;



TEST(GPU_manager_testing, constructor)
{

	GPU_manager gpm = GPU_manager(800,600);
	EXPECT_EQ(gpm.get_width(),800);
	EXPECT_EQ(gpm.get_height(),600);
}

TEST(GPU_manager_testing, free_buffers)
{
	GPU_manager gpm = GPU_manager(1024,768);
	gpm.free_internal_buffers();
	EXPECT_EQ(gpm.get_source_buffer(), nullptr);
	EXPECT_EQ(gpm.get_target_buffer(), nullptr);

}

TEST(GPU_manager_testing, get_source_buffer)
{
	GPU_manager gpm = GPU_manager(1920,1080);
	ASSERT_NE(gpm.get_source_buffer(), nullptr);

}

TEST(GPU_manager_testing, get_test_buffer)
{
	GPU_manager gpm = GPU_manager(1920,1080);
	ASSERT_NE(gpm.get_target_buffer(), nullptr);

}



//compute needed size of block grid kernel X
// allocate image  buffers on gpu X
// getter for gpu_buffers;X
// allocate generic buffers on gpu (example the one needed for the stancils)
// stancil function allocator passing directly a stancil pointer
//-- this will allow to simplfy the cuda call of filters, passing only
// the device buffers and size of the kernerl

//set getter grain_size
//getter grid dim
//getter block dim
//in the future be able to deal with SLI?