OBJDIR=obj
SRCDIR=src
BINDIR=bin
CPPFLAGS=-g -Wall -Wextra -std=c++14 -I $(SRCDIR) -I/opt/X11/include
CXX=g++
NVXX=nvcc
LIBS=-lX11 -lXext -lpthread -L/opt/X11/lib/
OBJS=$(OBJDIR)/canny.o
ifeq ($(HAS_CUDA),y)
	CUDA_LIBS=-lcudadevrt -lcudart -lcuda
else
	CUDA_LIBS=
endif
all: canny main

main: $(SRCDIR)/main.cpp
	$(CXX) $(SRCDIR)/main.cpp $(OBJS) -o $(BINDIR)/main $(CPPFLAGS) $(LIBS) $(CUDA_LIBS)

canny: $(SRCDIR)/canny.cpp
	$(CXX) -c $(SRCDIR)/canny.cpp -o $(OBJDIR)/canny.o $(CPPFLAGS)
