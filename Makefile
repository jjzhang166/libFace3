##MakeFile

CC := g++

LOCAL_PATH:= $(shell pwd)

LOCAL_STATIC_MACRO := -D CPU_ONLY=1 -D DLIB_JPEG_SUPPORT -D DLIB_PNG_SUPPORT

LOCAL_C_LIBRARIES := -L ${LOCAL_PATH} \
					-L ${LOCAL_PATH}/lib \
					-L ${LOCAL_PATH}/caffe/lib \
					-L ${LOCAL_PATH}/dlib-18.18 \


LOCAL_SHARE_LIB := -lcblas -lboost_system \
				-lopencv_core -lopencv_highgui -lopencv_imgproc \
				-lopencv_flann -lopencv_objdetect -lopencv_video\
				-lcaffe -ldlib -ljpeg -lpng -lglog \
				-pthread

INCLUDE := -I ${LOCAL_PATH}/include \
			-I ${LOCAL_PATH}/camera/include/ \
			-I ${LOCAL_PATH}/caffe/include \
			-I ${LOCAL_PATH}/dlib-18.18/ \
			-I /usr/include/opencv \
			-I /usr/include/opencv2/core

CPPFLAGS := 
CFLAGS :=
CXXFLAGS := -std=c++11 -g


SRC := ${LOCAL_PATH}/src
EXAM := ${LOCAL_PATH}/example

#VPATH = ./src:

target := $(shell pwd)
target := $(notdir $(target))
target := $(target).bin



srcs := $(shell ls $(SRC)/*.cpp)
objs := $(srcs:.cpp=.o)
#deps := $(srcs:.cpp=.d)

#include ../Makefile
#-include $(deps)
# include $(deps)相当与增加以下依赖
#1.o: 1.cpp 1.h
#2.o: 2.cpp
#3.o: 3.cpp
#4.o: 4.cpp
#5.o: 5.cpp


# 定义伪目标
.PHONY: all clean install uninstall

all: $(target)

%.o: %.cpp
	$(CC) -c -o $@ $< $(CPPFLAGS) $(CFLAGS) $(CXXFLAGS) $(INCLUDE) $(LOCAL_STATIC_MACRO)

$(target): $(objs)
	$(CC) -o $@ $^ $(LOCAL_SHARE_LIB) $(LOCAL_C_LIBRARIES) -Wl,--rpath=$(LOCAL_PATH)/camera/HCNet/lib:$(LOCAL_PATH)/camera/HCNet/HCNetSDKCom -Wl,--rpath=$(LOCAL_PATH)/caffe/lib 

clean:
	$(RM) $(SRC)/*.o
	$(RM) *.bin

#install:
#	$(CP) $(target) /usr/bin

#uninstall:
#        $(RM) /usr/bin/$(target)

#%.d: %.cpp
#	$(CC) -MM $< > $@
