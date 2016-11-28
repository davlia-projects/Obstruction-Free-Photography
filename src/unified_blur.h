#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "pipeline.h"

class UnifiedBlur: public Pipeline {
	private:
    int width;
    int height;
    uint8_t * dev_src;
    uint8_t * dev_dst;
	public:
		UnifiedBlur(int width, int height);
		~UnifiedBlur();
		int processFrame(uint8_t * frame);
		AVPixelFormat getPixelFormat();
};
