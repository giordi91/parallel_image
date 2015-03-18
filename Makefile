CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -W -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -std=c++0x -DLINUX -fPIE -m64
UIC = uic
TARGET = parallel_image
BUILD_PATH = build
SRC_PATH= src

INCLUDE_PATH = -I include -I /usr/local/cuda/include -isystem /opt/Qt/5.4/gcc_64/include
LIBS_PATH =  -L /opt/Qt/5.4/gcc_64/lib  -Wl,-rpath=/opt/Qt/Tools/QtCreator/lib/qtcreator
LIBS = -ltbb -lQt5Widgets -lQt5Test -lQt5Gui -lQt5Core

CUDA_LIB = -lcudart -lcudadevrt
CUDA_PATH = "/usr/local/cuda-6.5"
CUDA_LIB_PATH = -L /usr/local/cuda/lib64
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_FLAGS =  -arch=sm_35

#setup searching path
vpath %.cpp src/filters
vpath %.cpp src/ui
vpath %.h include/ui
vpath %.cpp src
vpath %.cpp src/core
vpath %.cu src/kernels
vpath %.ui src/ui/forms
vpath %.o build
vpath %.cpp build

.SUFFIXES: .cpp .o .cu .h


#list of object to build
OBJS = mainwindow.o    main.o bitmap.o bw_filter.o blur_filter.o stancil.o gaussian_filter.o convolution_filter.o sharpen_filter.o edge_detection_filter.o bw_kernel.cu.o blur_kernel.cu.o convolution_kernel.cu.o 
#object with added build path for linking purpose
F_OBJS = $(addprefix $(BUILD_PATH)/, $(OBJS))

UI_FORMS = ui_base_window.h

MOCS = moc_mainwindow.cpp

MOCS_OBJS = moc_mainwindow.o
F_MOCS_OBJS = $(addprefix $(BUILD_PATH)/, $(MOCS_OBJS))

all: $(UI_FORMS) $(MOCS) $(MOCS_OBJS) $(OBJS)
	$(CXX)  $(F_OBJS) $(F_MOCS_OBJS) -o $(BUILD_PATH)/$(TARGET) $(LIBS_PATH) $(LIBS) $(CUDA_LIB_PATH) $(CUDA_LIB)

%.o: %.cpp
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) -c $< -o $(BUILD_PATH)/$@

%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $(BUILD_PATH)/$@

run: clean all
	./build/parallel_image

	
clean:
	rm -f include/ui/ui_*.h
	rm -f $(BUILD_PATH)/*.o $(BUILD_PATH)/$(TARGET)* *.o

doc:
	rm -f -r ./doc/html
	doxygen ./Doxyfile
	google-chrome ./doc/html/index.html

ui_%.h: %.ui
	$(UIC) $< -o include/ui/$@

moc_%.cpp: %.h
	moc $< -o  build/$@


.PHONY: all run clean doc ui_compile moc_compile


#QT_PLUGIN_PATH=/opt/Qt/5.4/gcc_64/plugins/
