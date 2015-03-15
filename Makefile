CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -W -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -std=c++0x -DLINUX
TARGET = parallel_image
BUILD_PATH = build
INCLUDE_PATH = -I include -I /usr/local/cuda/include
SRC_PATH= src
CUDA_LIB = -L /usr/local/cuda/lib -lcudadevrt
CUDA_PATH = "/usr/local/cuda-6.5"
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_FLAGS =  -arch=sm_35

#setup searching path
vpath %.cpp src/filters
vpath %.cpp src
vpath %.cpp src/core
vpath %.cu src/kernels
vpath %.o build
 
#list of object to build
OBJS = main.o bitmap.o bw_filter.o blur_filter.o stancil.o gaussian_filter.o convolution_filter.o sharpen_filter.o edge_detection_filter.o bw_kernel.cu.o blur_kernel.cu.o convolution_kernel.cu.o
#object with added build path for linking purpose
F_OBJS = $(addprefix $(BUILD_PATH)/, $(OBJS))


all: $(OBJS)
	$(CXX)  $(F_OBJS) -o $(BUILD_PATH)/$(TARGET) -L /usr/local/cuda/lib64 -L /usr/local/lib  -ltbb  -lcudart

%.o: %.cpp
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) -c $< -o $(BUILD_PATH)/$@

%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $(BUILD_PATH)/$@

run: clean all

	$(BUILD_PATH)/$(TARGET) 

clean:
	echo $(F_OBJS)
	rm -f $(BUILD_PATH)/*.o $(BUILD_PATH)/$(TARGET)* *.o

doc:
	rm -f -r ./doc/html
	doxygen ./Doxyfile
	google-chrome ./doc/html/index.html

.PHONY: all run clean doc
