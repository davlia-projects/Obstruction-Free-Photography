#pragma once

#include "pipeline.h"

class Diff: public Pipeline {
  private:
    int width;
    int height;
    uint8_t * tempBuffer;
    uint8_t * prevImage;
    bool hasPrev;
  public:
    Diff(int width, int height);
    ~Diff();
    int processFrame(uint8_t * frame);
    AVPixelFormat getPixelFormat();
};
