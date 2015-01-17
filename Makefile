CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -W -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -std=c++0x -DLINUX
TARGET = parallel_image
BUILD_PATH = build
INCLUDE_PATH = -I include -I /usr/local/cuda/include
SRC_PATH= src
CUDA_LIB = -L /usr/local/cuda/lib -lcudadevrt
CUDA_PATH = "/usr/local/cuda-6.5"
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_FLAGS = -arch=sm_35

vpath %.cpp src

#TO DO , I want to build all the .o in the buil dir
# which I can do already but then I dunno how to read
# allthe .o properly for the $? symbol, need more 
#studies

all: main.o bitmap.o bw_filter.o blur_filter.o bw_kernel.o blur_kernel.o
	$(CXX)  $? -o $(BUILD_PATH)/$(TARGET) -L /usr/local/cuda/lib64 -L /usr/local/lib  -ltbb  -lcudart

bw_kernel.o: 
	$(NVCC) -c -arch=sm_35 ./src/bw_kernel.cu

blur_kernel.o: 
	$(NVCC) $(CUDA_FLAGS) -c ./src/blur_kernel.cu

%.o: %.cpp
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) -c $<

run: clean all
	$(BUILD_PATH)/$(TARGET) 

clean:
	rm -f $(BUILD_PATH)/*.o $(BUILD_PATH)/$(TARGET)* *.o

doc:
	rm -f -r ./doc/html
	doxygen ./Doxyfile
	google-chrome ./doc/html/index.html