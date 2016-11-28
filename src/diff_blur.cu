#include "diff_blur.h"
#include <thrust/remove.h>

__global__ void kernGenerateDiffs(int width, int height, DiffPoint * diffs, uint8_t * prev, uint8_t * img) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int z = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x >= width || y >= height || z >= 3) {
    return;
  }
	int idx = (y * width + x) * 3 + z;
	diffs[idx].x = x;
	diffs[idx].y = y;
  diffs[idx].c = z;
  diffs[idx].delta = (int)img[idx] - (int)prev[idx];
	diffs[idx].val = img[idx];
  return;
}

__global__ void kernBlurUsingDiffs(int N, int width, int height, float * prevblur, uint8_t * prev, DiffPoint * diffs) {
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
        atomicAdd(&prevblur[t], (float)d.delta * kernel[i][j]);
      }
    }
  }
	prev[(d.y * width + d.x) * 3 + d.c] = d.val;
  return;
}

__global__ void kernCopyToFrame(int N, uint8_t * frame, float * src) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= N) {
		return;
	}
	if (src[idx] < 0) {
		frame[idx] = 0;
	} else {
		frame[idx] = (uint8_t) src[idx];
	}
	return;
}

DiffBlur::DiffBlur(int width, int height) {
  this->width = width;
  this->height = height;
  int N = 3 * width * height;
  cudaMalloc(&this->dev_diffPoints, N * sizeof(DiffPoint));
	this->dev_thrust_diffPoints = thrust::device_pointer_cast(this->dev_diffPoints);
  cudaMalloc(&this->dev_prev, N * sizeof(uint8_t));
  cudaMemset(this->dev_prev, 0, N * sizeof(uint8_t));
  cudaMalloc(&this->dev_prevblur, N * sizeof(float));
  cudaMemset(this->dev_prevblur, 0, N * sizeof(float));
  cudaMalloc(&this->dev_frame, N * sizeof(uint8_t));
}

DiffBlur::~DiffBlur() {
  cudaFree(this->dev_diffPoints);
  cudaFree(this->dev_prev);
  cudaFree(this->dev_frame);
}

struct IsSmallDiff {
	const static int THRESHOLD = 25;
	__host__ __device__ bool operator()(const DiffPoint & d) {
		return (abs(d.delta) < THRESHOLD);
	}
};

int DiffBlur::processFrame(uint8_t * frame) {
  int N = 3 * this->width * this->height;
  const dim3 blockSize(256);
  const dim3 blocksPerGrid((N + blockSize.x - 1) / blockSize.x);

	const dim3 blockSize3d(8, 8, 3);
	const dim3 blocksPerGrid3d((this->width + blockSize3d.x - 1) / blockSize3d.x, (this->height + blockSize3d.y - 1) / blockSize3d.y, 1);
  
  cudaMemcpy(this->dev_frame, frame, N * sizeof(uint8_t), cudaMemcpyHostToDevice);
  kernGenerateDiffs<<<blocksPerGrid3d, blockSize3d>>>(this->width, this->height, this->dev_diffPoints, this->dev_prev, this->dev_frame);

	thrust::device_ptr<DiffPoint> dev_thrust_diffPoints_end =
		thrust::remove_if(this->dev_thrust_diffPoints, dev_thrust_diffPoints + N, IsSmallDiff());
	int n_diffs = dev_thrust_diffPoints_end - this->dev_thrust_diffPoints;

  kernBlurUsingDiffs<<<blocksPerGrid, blockSize>>>(n_diffs, this->width, this->height, this->dev_prevblur, this->dev_prev, this->dev_diffPoints);
	kernCopyToFrame<<<blocksPerGrid, blockSize>>>(N, this->dev_frame, this->dev_prevblur);
  cudaMemcpy(frame, this->dev_frame, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  return 0;

}

AVPixelFormat DiffBlur::getPixelFormat() {
  return AV_PIX_FMT_RGB24;
}
