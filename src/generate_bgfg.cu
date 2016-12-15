#include "generate_bgfg.h"
#include "timing.h"
#include <cuda.h>
#include <cstdio>

__global__
void kernTakeMean(int width, int height, float * img, unsigned char * orig, glm::vec2 offset, float SCALE, float * bgPixels, int frames, int curFrame) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
	int idx = y * width + x;
	glm::ivec2 warp = glm::ivec2(x + (int)offset.x, y + (int)offset.y);
	warp.x = max(0, min(width - 1, warp.x));
	warp.y = max(0, min(height - 1, warp.y));
	float t = (float)orig[warp.y * width + warp.x];
	img[idx] += SCALE * t;
	if (bgPixels != NULL) {
		bgPixels[frames * idx + curFrame] = t;
	}
	return;
}

void generateBgFg(int width, int height, int frames, float * bgImg, float * fgImg, unsigned char ** grayscale, std::pair<glm::vec2, glm::vec2> * groupVectors, float * bgPixels) {
	float * dev_bgImg, * dev_fgImg, * dev_bgPixels;
	unsigned char * dev_grayscale;
	int N = width * height;
	float SCALE = 1.0f / (float) (frames + 1);
  dim3 blockSize2d(16, 16);
  dim3 blocksPerGrid2d(
    (width + blockSize2d.x - 1) / blockSize2d.x,
    (height + blockSize2d.y - 1) / blockSize2d.y);
	cudaMalloc(&dev_bgPixels, (frames + 1) * N * sizeof(float));
	cudaMalloc(&dev_bgImg, N * sizeof(float));
	cudaMemset(dev_bgImg, 0, N * sizeof(float));
	cudaMalloc(&dev_fgImg, N * sizeof(float));
	cudaMemset(dev_fgImg, 0, N * sizeof(float));
	cudaMalloc(&dev_grayscale, N * sizeof(unsigned char));
	cudaMemcpy(dev_grayscale, grayscale[2], N * sizeof(unsigned char), cudaMemcpyHostToDevice);
  TIMEINIT
	TIMEIT((kernTakeMean<<<blocksPerGrid2d, blockSize2d>>>(width, height, dev_bgImg, dev_grayscale, glm::vec2(0.0f, 0.0f), SCALE, dev_bgPixels, frames + 1, 0)), "Take Mean")

	cudaMemcpy(dev_fgImg, dev_bgImg, N * sizeof(float), cudaMemcpyDeviceToDevice);
	for (int i = 0; i < frames; i++) {
		int ni = (i >= 2) ? i + 1 : i;
		cudaMemcpy(dev_grayscale, grayscale[ni], N * sizeof(unsigned char), cudaMemcpyHostToDevice);
		TIMEIT((kernTakeMean<<<blocksPerGrid2d, blockSize2d>>>(width, height, dev_bgImg, dev_grayscale, groupVectors[i].first, SCALE, dev_bgPixels, frames + 1, i + 1)), "Take Mean 1")
		TIMEIT((kernTakeMean<<<blocksPerGrid2d, blockSize2d>>>(width, height, dev_fgImg, dev_grayscale, groupVectors[i].second, SCALE, NULL, 0, 0)), "Take Mean 1")
		cudaDeviceSynchronize();
	}
  TIMEEND

	cudaMemcpy(bgImg, dev_bgImg, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(fgImg, dev_fgImg, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(bgPixels, dev_bgPixels, (frames + 1) * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_bgImg);
	cudaFree(dev_fgImg);
	cudaFree(dev_grayscale);
	cudaFree(dev_bgPixels);
	return;
}
