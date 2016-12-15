OBJDIR=obj
SRCDIR=src
BINDIR=bin
CPPFLAGS=-g -Wall -Wextra -std=c++14 -I $(SRCDIR) -I/opt/X11/include
CXX=g++
NVXX=nvcc
LIBS=-lX11 -lXext -lpthread -L/opt/X11/lib/
OBJS=$(OBJDIR)/canny.o $(OBJDIR)/flow.o $(OBJDIR)/lk.o $(OBJDIR)/kmeans.o $(OBJDIR)/interpolate.o

ifeq ($(HAS_CUDA),y)
	CUDA_LIBS=-lcudadevrt -lcudart -lcuda
else
	CUDA_LIBS=
endif
all: canny flow interpolate warp lk kmeans main

main: $(SRCDIR)/main.cpp
	$(CXX) $(SRCDIR)/main.cpp $(OBJS) -o $(BINDIR)/main $(CPPFLAGS) $(LIBS) $(CUDA_LIBS) `pkg-config opencv --cflags --libs`

canny: $(SRCDIR)/canny.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/canny.cu -o $(OBJDIR)/canny.o
endif
flow: $(SRCDIR)/flow.cpp
	$(CXX) -c $(SRCDIR)/flow.cpp -o $(OBJDIR)/flow.o $(CPPFLAGS)

lk: $(SRCDIR)/lk.cpp
	$(CXX) -c $(SRCDIR)/lk.cpp -o $(OBJDIR)/lk.o $(CPPFLAGS)

kmeans: $(SRCDIR)/kmeans.cpp
	$(CXX) -c $(SRCDIR)/kmeans.cpp -o $(OBJDIR)/kmeans.o $(CPPFLAGS)

interpolate: $(SRCDIR)/interpolate.cpp
	$(CXX) -c $(SRCDIR)/interpolate.cpp -o $(OBJDIR)/interpolate.o $(CPPFLAGS)

warp: $(SRCDIR)/warp.cpp
	$(CXX) -c $(SRCDIR)/warp.cpp -o $(OBJDIR)/warp.o $(CPPFLAGS)
