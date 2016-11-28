#pragma once

#include "pipeline.h"

class BasicBlur: public Pipeline {
  private:
    int width;
    int height;
    uint8_t * dev_src;
    uint8_t * dev_dst;

  public:
    BasicBlur(int width, int height);
    ~BasicBlur();
    int processFrame(uint8_t * frame);
    AVPixelFormat getPixelFormat();
};
