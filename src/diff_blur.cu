#include "diff_blur.h"

__global__ void kernGenerateDiffs(int width, int height, DiffPoint * diffs, uint8_t * prev, uint8_t * img) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= width * height * 3) {
    return;
  }
  diffs[idx].idx = idx;
  diffs[idx].val = img[idx];
  diffs[idx].delta = (int16_t)img[idx] - (int16_t)prev[idx];
  return;
}

__global__ void kernBlurUsingDiffs(int N, int width, int height, uint8_t * prev, DiffPoint * diffs) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= N) {
    return;
  }
  DiffPoint & d = diffs[idx];
  float kernel[5][5] = {
    {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
    {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
    {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
    {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
    {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
  };
  for (int i = 0; i < 5; i++) {
    int tx = d.x + i - 2;
    for (int j = 0; j < 5; j++) {
      int ty = d.y + j - 2;
      if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
        int t = (ty * width + tx) * 3 + d.c;
        atomicAdd(&prev[t], (uint8_t)((float)d.delta * kernel[i][j]));
      }
    }
  }
  return;
}

DiffBlur::DiffBlur(int width, int height) {
  this->width = width;
  this->height = height;
  int N = 3 * width * height;
  cudaMalloc(&this->dev_diffPoints, N * sizeof(DiffBlurPoint));
  cudaMalloc(&this->dev_prev, N * sizeof(uint8_t));
  cudaMemset(this->dev_prev, 0, N * sizeof(uint8_t));
  cudaMalloc(&this->dev_frame, N * sizeof(uint8_t));
}

DiffBlur::~DiffBlur() {
  cudaFree(this->dev_diffPoints);
  cudaFree(this->dev_prev);
  cudaFree(this->dev_frame);
}

int DiffBlur::processFrame(uint8_t * frame) {
  int N = 3 * this->width * this->height;
  const dim3 blockSize(256);
  const dim3 blocksPerGrid((N + blockSize.x - 1) / blockSize.x);
  
  cudaMemcpy(this->dev_frame, frame, N * sizeof(uint8_t), cudaMemcpyHostToDevice);
  kernGenerateDiffs<<<blocksPerGrid, blockSize>>>(this->width, this->height, this->dev_diffPoints, this->dev_prev, this->dev_frame);
  kernBlurUsingDiffs<<<blocksPerGrid, blockSize>>>(N, this->width, this->height, this->dev_prev, this->dev_diffPoints);
  cudaMemcpy(frame, this->dev_prev, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  return 0;

}

AVPixelFormat DiffBlur::getPixelFormat() {
  return AV_PIX_FMT_RGB24;
}
