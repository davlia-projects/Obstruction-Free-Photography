OBJDIR=obj
SRCDIR=src
BINDIR=bin
CPPFLAGS=-g -Wall -Wextra -std=c++11 -I $(SRCDIR)
CXX=g++
NVXX=nvcc
HAS_CUDA=y
ifeq ($(HAS_CUDA),y)
	CUDA_LIBS=-lcudadevrt -lcudart -lcuda
else
	CUDA_LIBS=
endif
LIBS=-lavutil -lavformat -lavcodec -lavdevice -lswscale

all: naive async basic unified blank diff diffblur main

main: $(SRCDIR)/main.cpp
	$(CXX) $(SRCDIR)/main.cpp $(SRCDIR)/pipeline.cpp $(OBJDIR)/diff_blur.o -o $(BINDIR)/main $(CPPFLAGS) $(LIBS) $(CUDA_LIBS)

blank: $(SRCDIR)/blank.cpp
	$(CXX) -c $(SRCDIR)/blank.cpp -o $(OBJDIR)/blank.o $(CPPFLAGS)

diff: $(SRCDIR)/diff.cpp
	$(CXX) -c $(SRCDIR)/diff.cpp -o $(OBJDIR)/diff.o $(CPPFLAGS)

naive: $(SRCDIR)/naive.cpp
	$(CXX) -c $(SRCDIR)/naive.cpp -o $(OBJDIR)/naive.o $(CPPFLAGS)

async: $(SRCDIR)/async_blur.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/async_blur.cu -o $(OBJDIR)/async_blur.o
endif

basic: $(SRCDIR)/basic_blur.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/basic_blur.cu -o $(OBJDIR)/basic_blur.o
endif

unified: $(SRCDIR)/unified_blur.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/unified_blur.cu -o $(OBJDIR)/unified_blur.o
endif

diffblur: $(SRCDIR)/diff_blur.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/diff_blur.cu -o $(OBJDIR)/diff_blur.o
endif

clean:
	rm bin/* obj/*
