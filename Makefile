OBJDIR=obj
SRCDIR=src
BINDIR=bin
CPPFLAGS=-g -Wall -Wextra -std=c++11 -I $(SRCDIR)
CXX=g++
NVXX=nvcc
HAS_CUDA=n
ifeq ($(HAS_CUDA),y)
	CUDA_LIBS=-lcudadevrt -lcudart -lcuda
else
	CUDA_LIBS=
endif
LIBS=-lavutil -lavformat -lavcodec -lavdevice -lswscale

all: naive async basic main

main: $(SRCDIR)/main.cpp
	$(CXX) $(SRCDIR)/main.cpp $(OBJDIR)/naive.o -o $(BINDIR)/main $(CPPFLAGS) $(LIBS) $(CUDA_LIBS)

naive: $(SRCDIR)/naive.cpp
	$(CXX) -c $(SRCDIR)/naive.cpp -o $(OBJDIR)/naive.o $(CPPFLAGS)

async: $(SRCDIR)/async_blur.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/async_blur.cu -o $(OBJDIR)/async_blur.o $(CPPFLAGS)
endif

basic: $(SRCDIR)/basic_blur.cu
ifeq ($(HAS_CUDA),y)
	$(CXX) -c $(SRCDIR)/basic_blur.cu -o $(OBJDIR)/basic_blur.o $(CPPFLAGS)
endif

clean:
	rm bin/* obj/*
