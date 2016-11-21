#include <cuda.h>
#include <stdint.h>
#include "basic_blur.h"

static uint8_t * dev_src[3];
static uint8_t * dev_dst[3];
static int cur = 0;
static cudaStream_t uploadStream, downloadStream, computeStream;

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
void init(int width, int height) {
	int sz = sizeof(uint8_t) * width * height * 3;
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&dev_src[i], sz);
		cudaMalloc(&dev_dst[i], sz);
	}
	cudaStreamCreate(&uploadStream);
	cudaStreamCreate(&computeStream);
	cudaStreamCreate(&downloadStream);
}

void cleanup() {
	for (int i = 0; i < 3; i++) {
		cudaFree(dev_src[i]);
		dev_src[i] = NULL;
		cudaFree(dev_dst[i]);
		dev_dst[i] = NULL;
	}
}

void blurFrame(uint8_t * dst, uint8_t * src, int width, int height) {
	int sz = sizeof(uint8_t) * width * height * 3;
	cudaMemcpyAsync(dev_src[cur % 3], src, sz, cudaMemcpyHostToDevice, uploadStream);
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(width + blockSize2d.x - 1) / blockSize2d.x,
		(height + blockSize2d.y - 1) / blockSize2d.y);
	
	if (cur >= 1) {
		kernGaussianBlur<<<blocksPerGrid2d, blockSize2d, 0, computeStream>>>(
			width, height, dev_dst[(cur - 1) % 3], dev_src[(cur + 1) % 3]);
	}
	if (cur >= 2) {
		cudaMemcpyAsync(dst, dev_dst[(cur - 2) % 3], sz, cudaMemcpyDeviceToHost, downloadStream);
	}
	cudaDeviceSynchronize();
	cur++;
	return;
}
