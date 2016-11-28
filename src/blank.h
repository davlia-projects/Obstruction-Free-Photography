#pragma once

#include "pipeline.h"

class BlankPipeline: public Pipeline {
  public:
    BlankPipeline();
    int processFrame(uint8_t * frame);
    AVPixelFormat getPixelFormat();
};
