CPPFLAGS=-g -Wall -Wextra -std=c++11
CXX=g++
NVXX=nvcc
LIBS=-lavutil -lavformat -lavcodec -lavdevice -lswscale

all:
	$(CXX) main.cpp naive.cpp -o main $(CPPFLAGS) $(LIBS)
