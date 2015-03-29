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
	EXPECT_NE(gpm.get_source_buffer(), nullptr);

}

TEST(GPU_manager_testing, get_test_buffer)
{
	GPU_manager gpm = GPU_manager(1920,1080);
	EXPECT_NE(gpm.get_target_buffer(), nullptr);

}

TEST(GPU_manager_testing, allocate_buffer)
{
	GPU_manager gpm = GPU_manager(1440,900);
	uint8_t * temp = nullptr;
	temp = gpm.allocate_device_buffer(1440,900,1);
	EXPECT_NE(temp, nullptr);
	cudaFree(temp);
}

TEST(GPU_manager_testing, init_and_get_grain_size)
{
	GPU_manager gpm = GPU_manager(1440,900);
	EXPECT_EQ(gpm.get_grain_size(),16);
	gpm = GPU_manager(1440,900,24);
	EXPECT_EQ(gpm.get_grain_size(),24);	
	EXPECT_THROW(GPU_manager(1440,900,999999999);,std::invalid_argument);
}

TEST(GPU_manager_testing, correct_block_size)
{
	GPU_manager gpm = GPU_manager(1440,900);
	dim3 bd = gpm.get_block_dim();
	EXPECT_EQ(bd.x , 16);
	EXPECT_EQ(bd.y ,16);	

	gpm = GPU_manager(1440,900,24);
	bd = gpm.get_block_dim();
	EXPECT_EQ(bd.x , 24);
	EXPECT_EQ(bd.y ,24);
}

TEST(GPU_manager_testing, correct_grid_size)
{
	size_t width = 1920;
	size_t height = 1080;
	size_t grain_size = 16;
	GPU_manager gpm = GPU_manager(width,height,grain_size);

	size_t width_blocks = ((width%grain_size) != 0)?(width/grain_size) +1: (width/grain_size);
    size_t height_blocks = ((height%grain_size) != 0)?(height/grain_size) +1: (height/grain_size);
   	
   	dim3 gd = gpm.get_grid_dim();

   	EXPECT_EQ(gd.x , width_blocks);
	EXPECT_EQ(gd.y ,height_blocks);


	grain_size = 32;
	gpm = GPU_manager(width,height,grain_size);
	
	width_blocks = ((width%grain_size) != 0)?(width/grain_size) +1: (width/grain_size);
    height_blocks = ((height%grain_size) != 0)?(height/grain_size) +1: (height/grain_size);
   	
    gd = gpm.get_grid_dim();

   	EXPECT_EQ(gd.x , width_blocks);
	EXPECT_EQ(gd.y ,height_blocks);

}




//compute needed size of block grid kernel X
//getter for gird/block dim 3 X
// allocate image  buffers on gpu X
// getter for gpu_buffers;X
// allocate generic buffers on gpu (example the one needed 
//for the stancils) X
// stancil function allocator passing directly a stancil pointer
//-- this will allow to simplfy the cuda call of filters, passing only
// the device buffers and size of the kernerl 

//setter grain size X 
//getter grain_size 
//getter grid dim X
//getter block dim X
//in the future be able to deal with SLI?