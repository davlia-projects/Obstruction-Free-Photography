#include <cuda.h>
#include <stdint.h>
#include "unified_blur.h"

__global__ void kernGaussianBlur(int width, int height, uint8_t * dst, uint8_t * src) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}
	float kernel[5][5] = {
    {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
    {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
    {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
    {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
    {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
  };
	float r, g, b;
	r = g = b = 0.0;
	for (int i = 0; i < 5; i++) {
		int tx = x + i - 2;
		for (int j = 0; j < 5; j++) {
			int ty = y + j - 2;
			if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
				r += src[(ty * width + tx) * 3] * kernel[i][j];
				g += src[(ty * width + tx) * 3 + 1] * kernel[i][j];
				b += src[(ty * width + tx) * 3 + 2] * kernel[i][j];
			}
		}
	}
	int idx = 3 * (y * width + x);
	dst[idx] = r;
	dst[idx + 1] = g;
	dst[idx + 2] = b;
	return;
}

UnifiedBlur::UnifiedBlur(int width, int height) {
  this->width = width;
  this->height = height;
  int sz = sizeof(uint8_t) * width * height * 3;
  cudaMallocManaged(&this->dev_src, sz);
  cudaMallocManaged(&this->dev_dst, sz);
}

UnifiedBlur::~UnifiedBlur() {
  cudaFree(this->dev_src);
  cudaFree(this->dev_dst);
}

AVPixelFormat UnifiedBlur::getPixelFormat() {
  return AV_PIX_FMT_RGB24;
}

int UnifiedBlur::processFrame(uint8_t * frame) {
  int sz = sizeof(uint8_t) * this->width * this->height * 3;
	memcpy(this->dev_src, frame, sz);

  const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(this->width + blockSize2d.x - 1) / blockSize2d.x,
		(this->height + blockSize2d.y - 1) / blockSize2d.y);
	
	kernGaussianBlur<<<blocksPerGrid2d, blockSize2d>>>(this->width, this->height, this->dev_dst, this->dev_src);
	cudaDeviceSynchronize();
	memcpy(frame, this->dev_dst, sz);
  return 0;
}

