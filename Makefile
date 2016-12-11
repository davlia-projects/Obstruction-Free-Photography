OBJDIR=obj
SRCDIR=src
BINDIR=bin
CPPFLAGS=-g -Wall -Wextra -std=c++14 -I $(SRCDIR) -I/opt/X11/include
CXX=g++
NVXX=nvcc
LIBS=-lX11 -lXext -lpthread -L/opt/X11/lib/
OBJS=$(OBJDIR)/canny.o $(OBJDIR)/flow.o $(OBJDIR)/interpolate.o
ifeq ($(HAS_CUDA),y)
	CUDA_LIBS=-lcudadevrt -lcudart -lcuda
else
	CUDA_LIBS=
endif
all: canny flow interpolate warp main

main: $(SRCDIR)/main.cpp
	$(CXX) $(SRCDIR)/main.cpp $(OBJS) -o $(BINDIR)/main $(CPPFLAGS) $(LIBS) $(CUDA_LIBS) `pkg-config opencv --cflags --libs`

canny: $(SRCDIR)/canny.cpp
	$(CXX) -c $(SRCDIR)/canny.cpp -o $(OBJDIR)/canny.o $(CPPFLAGS)

flow: $(SRCDIR)/flow.cpp
	$(CXX) -c $(SRCDIR)/flow.cpp -o $(OBJDIR)/flow.o $(CPPFLAGS)

interpolate: $(SRCDIR)/interpolate.cpp
	$(CXX) -c $(SRCDIR)/interpolate.cpp -o $(OBJDIR)/interpolate.o $(CPPFLAGS)

warp: $(SRCDIR)/warp.cpp
	$(CXX) -c $(SRCDIR)/warp.cpp -o $(OBJDIR)/warp.o $(CPPFLAGS)
