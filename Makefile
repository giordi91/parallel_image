CURR_OS = NOT_SET
DIST = NOT_SET

##########################################
#Checking for OS
##########################################

#checking linux 
#checking windows
ifeq ($(OS), Windows_NT)
CURR_OS = Windows

else
endif
OSTYPE := $(shell uname -s)
ifeq ($(OSTYPE), Linux)
CURR_OS = linux
TEMP_ID = $(shell cat /etc/*-release)
#checking which linux distro
ifneq (, $(findstring CentOS, $(TEMP_ID)))
	DIST = CentOS
endif

ifneq (, $(findstring Ubuntu, $(TEMP_ID)))
	DIST = Ubuntu
endif


endif

include cuda_config 

CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -W -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -std=c++0x -DLINUX
TARGET = parallel_image
BUILD_PATH = build
INCLUDE_PATH = -I include -I /usr/local/cuda/include
SRC_PATH= src
CUDA_LIB = -L /usr/local/cuda/lib -lcudadevrt
CUDA_PATH = "/usr/local/cuda-6.5"
NVCC = $(CUDA_PATH)/bin/nvcc


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
	$(NVCC) -c -arch=sm_35 ./src/blur_kernel.cu

%.o: %.cpp
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) -c $<

run: clean all
	$(BUILD_PATH)/$(TARGET) 

clean:
	rm -f $(BUILD_PATH)/*.o $(BUILD_PATH)/$(TARGET)* *.o


#############################################
# Verbose OS
#############################################
.PHONY: verbose doc
	
verbose:

	@echo Build Started ....
	@echo -----Environment-------
	@echo Operating System: $(CURR_OS)
	@echo Operating System Distribution: $(DIST)
	@echo -----------------------

doc:
	rm -f -r ./doc/html
	doxygen ./Doxyfile
	google-chrome ./doc/html/index.html