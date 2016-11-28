#pragma once

#include "pipeline.h"

struct DiffPoint {
  int x;
  int y;
  int c;
  float val;
  int16_t delta;
};

class DiffBlur: public Pipeline {
  private:
    int width;
    int height;
    DiffPoint * dev_diffPoints;
    uint8_t * dev_prev;
    uint8_t * dev_frame;
  public:
    DiffBlur(int width, int height);
    ~DiffBlur();
    int processFrame(uint8_t * frame);
    AVPixelFormat getPixelFormat();
};
