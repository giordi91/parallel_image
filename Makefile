CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -W -Wconversion -Wshadow -Wcast-qual -Wwrite-strings
TARGET = parallel_image
BUILD_PATH = ./build
SRC_PATH= ./src
INCLUDE_PATH = ./include

all: main.cpp
	$(CXX) $(CXXFLAGS) -L . -I $(INCLUDE_PATH)  $(SRC_PATH)/main.cpp -o $(BUILD_PATH)/$(TARGET) 

main.cpp:

run:
	$(BUILD_PATH)/$(TARGET) 

clean:
	rm -f $(BUILD_PATH)/*.o $(BUILD_PATH)/$(TARGET)

