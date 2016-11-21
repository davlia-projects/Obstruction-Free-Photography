CPPFLAGS=-g -Wall -Wextra -std=c++11
CXX=g++
NVXX=nvcc
LIBS=-lavutil -lavformat -lavcodec -lavdevice -lswscale -lcudadevrt -lcudart -lcuda

all:
	$(NVXX) -c basic_blur.cu -o basic_blur.o
	$(CXX) main.cpp basic_blur.o -o main $(CPPFLAGS) $(LIBS)
