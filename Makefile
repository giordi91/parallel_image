CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -W -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -std=c++11 -DLINUX -fPIE -m64
UIC = uic
TARGET = parallel_image
TEST_TARGET = parallel_test
BUILD_PATH = build
SRC_PATH= src

INCLUDE_PATH = -I include -I /usr/local/cuda/include  \
				-isystem /opt/Qt/5.4/gcc_64/include \
				-isystem /home/giordi/WORK_IN_PROGRESS/C/libs/googletest/include \
				-isystem /home/giordi/WORK_IN_PROGRESS/C/libs/googlemock/include \
				-isystem middleware

LIBS_PATH =  -L /opt/Qt/5.4/gcc_64/lib  -Wl,-rpath=/opt/Qt/Tools/QtCreator/lib/qtcreator
LIBS = -ltbb -lQt5Widgets -lQt5Test -lQt5Gui -lQt5Core

CUDA_LIB = -lcudart -lcudadevrt
CUDA_PATH = "/usr/local/cuda-6.5"
CUDA_LIB_PATH = -L /usr/local/cuda/lib64
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_FLAGS =  -arch=sm_35


#tests 
TEST_LIB_PATH = -L /home/giordi/WORK_IN_PROGRESS/C/libs/lib
TEST_LIB = -lgmock -lgmock_main

#setup searching path
#regular cpp source
vpath %.cpp src src/core src/filters src/ui tests
#ui file source
vpath %.h include/ui
vpath %.ui src/ui/forms
#cuda file source
vpath %.cu src/kernels
#final obj source
vpath %.o build
#oatg fir tge tests


.SUFFIXES: .cpp .o .cu .h


#list of object to build
OBJS = mainwindow.o texturewidget.o filter_widget.o glsl_program.o glsl_shader.o main.o bitmap.o bw_filter.o stancil.o convolution.o \
	   convolution_filter.o gaussian_filter.o  sharpen_filter.o  \
	   edge_detection_filter.o bw_kernel.cu.o convolution_kernel.cu.o \
	   filter_manager.o GPU_manager.o

#object with added build path for linking purpose
F_OBJS = $(addprefix $(BUILD_PATH)/, $(OBJS))
#the ui file we need to generate
UI_FORMS = ui_base_window.h ui_filter_widget.h
#the moc files we need to extract from header
MOCS = moc_mainwindow.cpp moc_texturewidget.cpp moc_filter_widget.cpp
#the moc objs we need to generate from the correspectviev moc*.cpp
MOCS_OBJS = moc_mainwindow.o moc_texturewidget.o moc_filter_widget.o
#the final moc objects that need to be linked
F_MOCS_OBJS = $(addprefix $(BUILD_PATH)/, $(MOCS_OBJS))

TEST_OBJS = filter_manager_test.o filter_manager.o \
			bitmap.o \
			GPU_manager.o gpu_manager_test.o \
			bw_filter.o bw_kernel.cu.o \
			edge_detection_filter.o \
			sharpen_filter.o \
			gaussian_filter.o \
			convolution_filter.o convolution_kernel.cu.o convolution.o \
			stancil.o \
			attribute_tests.o \
			ui_tests.o mainwindow.o texturewidget.o moc_texturewidget.o \
			glsl_program.o glsl_shader.o moc_mainwindow.o


# TEST_OBJS :=  $(filter-out mainwindow.o main.o,$(OBJS))
F_TEST_OBJS = $(addprefix $(BUILD_PATH)/, $(TEST_OBJS))

all: ui $(MOCS_OBJS) $(OBJS) 
	$(CXX)  $(F_OBJS) $(F_MOCS_OBJS) -o $(BUILD_PATH)/$(TARGET) $(LIBS_PATH) $(LIBS) $(CUDA_LIB_PATH) $(CUDA_LIB)

run: clean all
	./$(BUILD_PATH)/$(TARGET) 

clean:
	rm -f data/OUT_*.bmp
	rm -f include/ui/ui_*.h build/moc*.cpp
	rm -f $(BUILD_PATH)/*.o $(BUILD_PATH)/$(TARGET)* \
	 						$(BUILD_PATH)/$(TEST_TARGET)* *.o

doc:
	rm -f -r ./doc/html
	doxygen ./Doxyfile
	google-chrome ./doc/html/index.html

tests:$(UI_FORMS) $(MOCS) $(MOCS_OBJS) $(TEST_OBJS)
	$(CXX)  $(F_TEST_OBJS) -o $(BUILD_PATH)/$(TEST_TARGET) \
	$(LIBS_PATH) $(LIBS) \
	$(CUDA_LIB_PATH) $(CUDA_LIB) \
	$(TEST_LIB_PATH) $(TEST_LIB)

	./$(BUILD_PATH)/$(TEST_TARGET) --gtest_filter=-Heavy* 
	
tests_heavy:$(UI_FORMS) $(MOCS) $(MOCS_OBJS) $(TEST_OBJS)
	$(CXX)  $(F_TEST_OBJS) -o $(BUILD_PATH)/$(TEST_TARGET) \
	$(LIBS_PATH) $(LIBS) \
	$(CUDA_LIB_PATH) $(CUDA_LIB) \
	$(TEST_LIB_PATH) $(TEST_LIB)

	./$(BUILD_PATH)/$(TEST_TARGET) --gtest_filter=Heavy* 

tests_all:$(UI_FORMS) $(MOCS) $(MOCS_OBJS) $(TEST_OBJS)
	$(CXX)  $(F_TEST_OBJS) -o $(BUILD_PATH)/$(TEST_TARGET) \
	$(LIBS_PATH) $(LIBS) \
	$(CUDA_LIB_PATH) $(CUDA_LIB) \
	$(TEST_LIB_PATH) $(TEST_LIB)

	./$(BUILD_PATH)/$(TEST_TARGET)


ui_%.h: %.ui
	$(UIC) $< -o include/ui/$@

moc_%.cpp: %.h
	moc $< -o  $(BUILD_PATH)/$@

moc_%.o: moc_%.cpp
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) -c build/$< -o $(BUILD_PATH)/$@

%.o: %.cpp
	$(CXX) $(CXXFLAGS)  $(INCLUDE_PATH) -c $< -o $(BUILD_PATH)/$@

%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $(BUILD_PATH)/$@

ui: $(UI_FORMS) $(MOCS) 



.PHONY: all run clean doc tests ui

#QT_PLUGIN_PATH=/opt/Qt/5.4/gcc_64/plugins/
