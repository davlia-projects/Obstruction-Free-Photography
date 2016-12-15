OBJDIR=obj
SRCDIR=src
BINDIR=bin
CPPFLAGS=-g -Wall -Wextra -std=c++11 -I $(SRCDIR) -I/opt/X11/include -I/opt/cuda/include
CXX=g++
NVXX=nvcc
LIBS=-lX11 -lXext -lpthread -L/opt/X11/lib/ -lavutil -lavformat -lavcodec -lavdevice -lswscale
HAS_CUDA=y
OBJS=$(OBJDIR)/canny.o $(OBJDIR)/flow.o $(OBJDIR)/lk.o $(OBJDIR)/ransac.o $(OBJDIR)/interpolate.o $(OBJDIR)/gradient_descent.o

ifeq ($(HAS_CUDA),y)
	CUDA_LIBS=-lcudadevrt -lcudart -lcuda
else
	CUDA_LIBS=
endif
all: canny flow warp lk ransac interpolate gd main 

main: $(SRCDIR)/main.cpp
	$(CXX) $(SRCDIR)/main.cpp $(OBJS) -o $(BINDIR)/main $(CPPFLAGS) $(LIBS) $(CUDA_LIBS) `pkg-config opencv --cflags --libs`

canny: $(SRCDIR)/canny.cpp
	$(CXX) -c $(SRCDIR)/canny.cpp -o $(OBJDIR)/canny.o $(CPPFLAGS)

flow: $(SRCDIR)/flow.cpp
	$(CXX) -c $(SRCDIR)/flow.cpp -o $(OBJDIR)/flow.o $(CPPFLAGS)

lk: $(SRCDIR)/lk.cpp
	$(CXX) -c $(SRCDIR)/lk.cpp -o $(OBJDIR)/lk.o $(CPPFLAGS)

ransac: $(SRCDIR)/kmeans.cpp
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/ransac.cu -o $(OBJDIR)/ransac.o -std=c++11
else
	$(CXX) -c $(SRCDIR)/kmeans.cpp -o $(OBJDIR)/ransac.o $(CPPFLAGS)
endif

interpolate: $(SRCDIR)/interpolate.cpp
	$(CXX) -c $(SRCDIR)/interpolate.cpp -o $(OBJDIR)/interpolate.o $(CPPFLAGS)

warp: $(SRCDIR)/warp.cpp
	$(CXX) -c $(SRCDIR)/warp.cpp -o $(OBJDIR)/warp.o $(CPPFLAGS)

gd: $(SRCDIR)/gradient_descent.cpp
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/gpu_gradient_descent.cu -o $(OBJDIR)/gradient_descent.o
else
	$(CXX) -c $(SRCDIR)/gradient_descent.cpp -o $(OBJDIR)/gradient_descent.o $(CPPFLAGS)
endif
