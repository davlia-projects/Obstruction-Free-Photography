CPPFLAGS=-g -Wall -Wextra -std=c++11
CXX=g++
NVXX=nvcc
LIBS=-lavutil -lavformat -lavcodec -lavdevice -lswscale -lcudadevrt -lcudart -lcuda

all:
	$(NVXX) -c async_blur.cu -o async_blur.o
	$(CXX) main.cpp async_blur.o -o main $(CPPFLAGS) $(LIBS)
