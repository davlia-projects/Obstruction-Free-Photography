#include "spatial_coherence.h"
#include <glm/glm.hpp>
#include <cuda.h>

__global__
void kernSpatialCoherence(int width, int height, int frames, float * img_cur, float * img_next, unsigned char * orig, float * pixels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
	int idx = y * width + x;
	float mn = 1e12;
	float val = img_cur[idx];
	for (int i = 0; i < frames; i++) {
		float d = 0.0f;
		for (int dy = -3; dy <= 3; dy++) {
			for (int dx = -3; dx <= 3; dx++) {
				int nx = glm::clamp(x + dx, 0, width - 1);
				int ny = glm::clamp(y + dy, 0, height - 1);
				int nidx = ny * width + nx;
				d += glm::abs(pixels[idx * frames + i] - img_cur[nidx]);
			}
		}
		d += 25.0f * glm::abs(pixels[idx * frames + i] - (float)orig[idx]);
		if (d < mn) {
			mn = d;
			val = pixels[idx * frames + i];
		}
	}
	img_next[idx] = val;
	return;
}

void spatialCoherence(int width, int height, int frames, float * img, unsigned char * orig, float * pixels) {
	int N = width * height;
	float * dev_img[2];
	unsigned char * dev_orig;
	float * dev_pixels;
	int cur = 0;
	for (int i = 0; i < 2; i++) {
		cudaMalloc(&dev_img[i], N * sizeof(float));
	}
	cudaMalloc(&dev_orig, N * sizeof(unsigned char));
	cudaMalloc(&dev_pixels, frames * N * sizeof(float));
	cudaMemcpy(dev_img[0], img, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_orig, orig, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pixels, pixels, frames * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize2d(16, 16);
  dim3 blocksPerGrid2d(
    (width + blockSize2d.x - 1) / blockSize2d.x,
    (height + blockSize2d.y - 1) / blockSize2d.y);

	for (int i = 0; i < 10; i++) {
		kernSpatialCoherence<<<blocksPerGrid2d, blockSize2d>>>(width, height, frames, dev_img[cur], dev_img[1 - cur], dev_orig, dev_pixels);
		cur = 1 - cur;
	}
	cudaMemcpy(img, dev_img[cur], N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_orig);
	cudaFree(dev_img[0]);
	cudaFree(dev_img[1]);
	cudaFree(dev_pixels);
	return;
}
