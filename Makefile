OS = NOT_SET
DIST = NOT_SET

##########################################
#Checking for OS
##########################################

#checking linux 
ifeq ($(OSTYPE), linux)
OS = linux
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
ifeq ($(OSTYPE), Windows)
OS = Windows
endif

endif


CXX = g++
CXXFLAGS = -ansi -pedantic -Wall -W -Wconversion -Wshadow -Wcast-qual -Wwrite-strings
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
	rm -f $(BUILD_PATH)/*.o $(BUILD_PATH)/$(TARGET)


#############################################
# Verbose OS
#############################################
.verbose:
	@echo 
	@echo Build Started ....
	@echo -----Environment-------
	@echo Operating System: $(OS)
	@echo Operating System Distribution: $(DIST)
	@echo -----------------------
