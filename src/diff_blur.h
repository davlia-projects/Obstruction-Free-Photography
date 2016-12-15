#pragma once

#include "pipeline.h"
#include <thrust/device_ptr.h>

struct DiffPoint {
  int x;
  int y;
  int c;
	uint8_t val;
  int delta;
};

class DiffBlur: public Pipeline {
  private:
    int width;
    int height;
    DiffPoint * dev_diffPoints;
    uint8_t * dev_prev;
    float * dev_prevblur;
    uint8_t * dev_frame;
		thrust::device_ptr<DiffPoint> dev_thrust_diffPoints;

		int kernSize;
		float * dev_kernel;
  public:
    DiffBlur(int width, int height);
    ~DiffBlur();
    int processFrame(uint8_t * frame);
    AVPixelFormat getPixelFormat();
};
