#CURR_OS = NOT_SET
DIST = NOT_SET

##########################################
#Checking for OS
##########################################

#checking linux 

ifeq ($(OSTYPE), linux)
CURR_OS = linux
TEMP_ID = $(shell cat /etc/*-release)
#checking which linux distro
ifneq (, $(findstring CentOS, $(TEMP_ID)))
	DIST = CentOS
endif

ifneq (, $(findstring Ubuntu, $(TEMP_ID)))
	DIST = Ubuntu
endif

else

#checking windows
ifeq ($(OS), Windows_NT)
CURR_OS = Windows
endif

endif


CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -W -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -std=c++0x
TARGET = parallel_image
BUILD_PATH = ./build
SRC_PATH= ./src
INCLUDE_PATH = ./include

all: .verbose main.cpp 
	$(CXX) $(CXXFLAGS) -L . -I $(INCLUDE_PATH)  $(SRC_PATH)/main.cpp -o $(BUILD_PATH)/$(TARGET) 

main.cpp:

run:
	$(BUILD_PATH)/$(TARGET) 

clean:
	rm -f $(BUILD_PATH)/*.o $(BUILD_PATH)/$(TARGET)*


#############################################
# Verbose OS
#############################################
.verbose:
	@echo Build Started ....
	@echo -----Environment-------
	@echo Operating System: $(CURR_OS)
	@echo Operating System Distribution: $(DIST)
	@echo -----------------------
