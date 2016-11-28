#pragma once

#include "pipeline.h"

class NaiveBlur: public Pipeline {
  private:
    int width;
    int height;
    uint8_t * tempBuffer;
  public:
    NaiveBlur(int width, int height);
    int processFrame(uint8_t * frame);
    AVPixelFormat getPixelFormat();
};
