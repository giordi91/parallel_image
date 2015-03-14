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
#-Xptxas --warning-as-error
vpath %.cpp src
vpath %.cu src
# VPATH = src
 
#TO DO , I want to build all the .o in the buil dir
# which I can do already but then I dunno how to read
# allthe .o properly for the $? symbol, need more 
#studies

.SUFFIXES: .cpp .o .cu .h

all: main.o bitmap.o bw_filter.o blur_filter.o stancil.o gaussian_stancil.o convolution_filter.o sharpen_filter.o bw_kernel.cu.o blur_kernel.cu.o convolution_kernel.cu.o
	$(CXX)  $? -o $(BUILD_PATH)/$(TARGET) -L /usr/local/cuda/lib64 -L /usr/local/lib  -ltbb  -lcudart


%.o: %.cpp
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) -c $<

%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

run: clean all
	$(BUILD_PATH)/$(TARGET) 

clean:
	rm -f $(BUILD_PATH)/*.o $(BUILD_PATH)/$(TARGET)* *.o

doc:
	rm -f -r ./doc/html
	doxygen ./Doxyfile
	google-chrome ./doc/html/index.html

.PHONY: all run clean
