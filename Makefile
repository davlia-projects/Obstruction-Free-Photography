OBJDIR=obj
SRCDIR=src
BINDIR=bin
CPPFLAGS=-g -Wall -Wextra -std=c++11 -I $(SRCDIR) -I/opt/X11/include -I/opt/cuda/include
CXX=g++
NVXX=nvcc
LIBS=-lX11 -lXext -lpthread -L/opt/X11/lib/ -lavutil -lavformat -lavcodec -lavdevice -lswscale
HAS_CUDA=y
OBJS=$(OBJDIR)/canny.o $(OBJDIR)/flow.o $(OBJDIR)/lk.o $(OBJDIR)/ransac.o $(OBJDIR)/interpolate.o $(OBJDIR)/gradient_descent.o $(OBJDIR)/generate_bgfg.o $(OBJDIR)/spatial_coherence.o
CPU_OBJS=$(OBJDIR)/ransac_cpu.o $(OBJDIR)/gd_cpu.o

ifeq ($(HAS_CUDA),y)
	CUDA_LIBS=-lcudadevrt -lcudart -lcuda
else
	CUDA_LIBS=
endif
all: canny flow warp lk ransac interpolate gd generate_bgfg main 

main: $(SRCDIR)/main.cpp
	$(CXX) $(SRCDIR)/main.cpp $(OBJS) $(CPU_OBJS) -o $(BINDIR)/main $(CPPFLAGS) $(LIBS) $(CUDA_LIBS) `pkg-config opencv --cflags --libs`

main2: $(SRCDIR)/main.cpp
	$(CXX) $(SRCDIR)/main2.cpp $(OBJS) -o $(BINDIR)/main2 $(CPPFLAGS) $(LIBS) $(CUDA_LIBS) `pkg-config opencv --cflags --libs`

canny: $(SRCDIR)/canny.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/canny.cu -o $(OBJDIR)/canny.o -std=c++11
endif

canny_cpu: $(SRCDIR)/canny.cpp
	$(CXX) -c $(SRCDIR)/canny.cpp -o $(OBJDIR)/canny.o $(CPPFLAGS)

flow: $(SRCDIR)/flow.cpp
	$(CXX) -c $(SRCDIR)/flow.cpp -o $(OBJDIR)/flow.o $(CPPFLAGS)

lk: $(SRCDIR)/lk.cpp
	$(CXX) -c $(SRCDIR)/lk.cpp -o $(OBJDIR)/lk.o $(CPPFLAGS)

ransac: $(SRCDIR)/ransac.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/ransac.cu -o $(OBJDIR)/ransac.o -std=c++11
else
	$(CXX) -c $(SRCDIR)/kmeans.cpp -o $(OBJDIR)/ransac.o $(CPPFLAGS)
endif

ransac_cpu: $(SRCDIR)/kmeans.cpp
	$(CXX) -c $(SRCDIR)/kmeans.cpp -o $(OBJDIR)/ransac_cpu.o $(CPPFLAGS)

interpolate: $(SRCDIR)/interpolate.cpp
	$(CXX) -c $(SRCDIR)/interpolate.cpp -o $(OBJDIR)/interpolate.o $(CPPFLAGS)

warp: $(SRCDIR)/warp.cpp
	$(CXX) -c $(SRCDIR)/warp.cpp -o $(OBJDIR)/warp.o $(CPPFLAGS)

gd: $(SRCDIR)/gradient_descent.cpp
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/gpu_gradient_descent.cu -o $(OBJDIR)/gradient_descent.o -std=c++11
else
	$(CXX) -c $(SRCDIR)/gradient_descent.cpp -o $(OBJDIR)/gradient_descent.o $(CPPFLAGS)
endif

gd_cpu: $(SRCDIR)/gradient_descent.cpp
	$(CXX) -c $(SRCDIR)/gradient_descent.cpp -o $(OBJDIR)/gd_cpu.o $(CPPFLAGS)

generate_bgfg: $(SRCDIR)/generate_bgfg.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/generate_bgfg.cu -o $(OBJDIR)/generate_bgfg.o -std=c++11
endif

spatial_coherence: $(SRCDIR)/spatial_coherence.cu
ifeq ($(HAS_CUDA),y)
	$(NVXX) -c $(SRCDIR)/spatial_coherence.cu -o $(OBJDIR)/spatial_coherence.o -std=c++11
endif
