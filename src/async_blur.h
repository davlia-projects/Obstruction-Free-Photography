#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "pipeline.h"

class AsyncBlur: public Pipeline {
  private:
    int width;
    int height;
    uint8_t * dev_src[3];
    uint8_t * dev_dst[3];
    uint8_t * tmp_dst;
    cudaStream_t uploadStream;
    cudaStream_t downloadStream;
    cudaStream_t computeStream;
    int cur;

  public:
    AsyncBlur(int width, int height);
    ~AsyncBlur();
    int processFrame(uint8_t * frame);
    AVPixelFormat getPixelFormat();
};
