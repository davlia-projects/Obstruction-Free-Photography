#include "diff_blur.h"
#include <thrust/remove.h>

const float kernel[21 * 21] = {
0.002179, 0.002192, 0.002204, 0.002214, 0.002223, 0.002231, 0.002237, 0.002242, 0.002246, 0.002248, 0.002248, 0.002248, 0.002246, 0.002242, 0.002237, 0.002231, 0.002223, 0.002214, 0.002204, 0.002192, 0.002179,
0.002192, 0.002205, 0.002217, 0.002228, 0.002237, 0.002244, 0.002251, 0.002256, 0.002259, 0.002261, 0.002262, 0.002261, 0.002259, 0.002256, 0.002251, 0.002244, 0.002237, 0.002228, 0.002217, 0.002205, 0.002192,
0.002204, 0.002217, 0.002229, 0.002239, 0.002249, 0.002256, 0.002263, 0.002268, 0.002271, 0.002273, 0.002274, 0.002273, 0.002271, 0.002268, 0.002263, 0.002256, 0.002249, 0.002239, 0.002229, 0.002217, 0.002204,
0.002214, 0.002228, 0.002239, 0.00225, 0.002259, 0.002267, 0.002273, 0.002278, 0.002282, 0.002284, 0.002285, 0.002284, 0.002282, 0.002278, 0.002273, 0.002267, 0.002259, 0.00225, 0.002239, 0.002228, 0.002214,
0.002223, 0.002237, 0.002249, 0.002259, 0.002268, 0.002276, 0.002282, 0.002287, 0.002291, 0.002293, 0.002294, 0.002293, 0.002291, 0.002287, 0.002282, 0.002276, 0.002268, 0.002259, 0.002249, 0.002237, 0.002223,
0.002231, 0.002244, 0.002256, 0.002267, 0.002276, 0.002284, 0.00229, 0.002295, 0.002299, 0.002301, 0.002302, 0.002301, 0.002299, 0.002295, 0.00229, 0.002284, 0.002276, 0.002267, 0.002256, 0.002244, 0.002231,
0.002237, 0.002251, 0.002263, 0.002273, 0.002282, 0.00229, 0.002297, 0.002302, 0.002305, 0.002308, 0.002308, 0.002308, 0.002305, 0.002302, 0.002297, 0.00229, 0.002282, 0.002273, 0.002263, 0.002251, 0.002237,
0.002242, 0.002256, 0.002268, 0.002278, 0.002287, 0.002295, 0.002302, 0.002307, 0.00231, 0.002313, 0.002313, 0.002313, 0.00231, 0.002307, 0.002302, 0.002295, 0.002287, 0.002278, 0.002268, 0.002256, 0.002242,
0.002246, 0.002259, 0.002271, 0.002282, 0.002291, 0.002299, 0.002305, 0.00231, 0.002314, 0.002316, 0.002317, 0.002316, 0.002314, 0.00231, 0.002305, 0.002299, 0.002291, 0.002282, 0.002271, 0.002259, 0.002246,
0.002248, 0.002261, 0.002273, 0.002284, 0.002293, 0.002301, 0.002308, 0.002313, 0.002316, 0.002318, 0.002319, 0.002318, 0.002316, 0.002313, 0.002308, 0.002301, 0.002293, 0.002284, 0.002273, 0.002261, 0.002248,
0.002248, 0.002262, 0.002274, 0.002285, 0.002294, 0.002302, 0.002308, 0.002313, 0.002317, 0.002319, 0.00232, 0.002319, 0.002317, 0.002313, 0.002308, 0.002302, 0.002294, 0.002285, 0.002274, 0.002262, 0.002248,
0.002248, 0.002261, 0.002273, 0.002284, 0.002293, 0.002301, 0.002308, 0.002313, 0.002316, 0.002318, 0.002319, 0.002318, 0.002316, 0.002313, 0.002308, 0.002301, 0.002293, 0.002284, 0.002273, 0.002261, 0.002248,
0.002246, 0.002259, 0.002271, 0.002282, 0.002291, 0.002299, 0.002305, 0.00231, 0.002314, 0.002316, 0.002317, 0.002316, 0.002314, 0.00231, 0.002305, 0.002299, 0.002291, 0.002282, 0.002271, 0.002259, 0.002246,
0.002242, 0.002256, 0.002268, 0.002278, 0.002287, 0.002295, 0.002302, 0.002307, 0.00231, 0.002313, 0.002313, 0.002313, 0.00231, 0.002307, 0.002302, 0.002295, 0.002287, 0.002278, 0.002268, 0.002256, 0.002242,
0.002237, 0.002251, 0.002263, 0.002273, 0.002282, 0.00229, 0.002297, 0.002302, 0.002305, 0.002308, 0.002308, 0.002308, 0.002305, 0.002302, 0.002297, 0.00229, 0.002282, 0.002273, 0.002263, 0.002251, 0.002237,
0.002231, 0.002244, 0.002256, 0.002267, 0.002276, 0.002284, 0.00229, 0.002295, 0.002299, 0.002301, 0.002302, 0.002301, 0.002299, 0.002295, 0.00229, 0.002284, 0.002276, 0.002267, 0.002256, 0.002244, 0.002231,
0.002223, 0.002237, 0.002249, 0.002259, 0.002268, 0.002276, 0.002282, 0.002287, 0.002291, 0.002293, 0.002294, 0.002293, 0.002291, 0.002287, 0.002282, 0.002276, 0.002268, 0.002259, 0.002249, 0.002237, 0.002223,
0.002214, 0.002228, 0.002239, 0.00225, 0.002259, 0.002267, 0.002273, 0.002278, 0.002282, 0.002284, 0.002285, 0.002284, 0.002282, 0.002278, 0.002273, 0.002267, 0.002259, 0.00225, 0.002239, 0.002228, 0.002214,
0.002204, 0.002217, 0.002229, 0.002239, 0.002249, 0.002256, 0.002263, 0.002268, 0.002271, 0.002273, 0.002274, 0.002273, 0.002271, 0.002268, 0.002263, 0.002256, 0.002249, 0.002239, 0.002229, 0.002217, 0.002204,
0.002192, 0.002205, 0.002217, 0.002228, 0.002237, 0.002244, 0.002251, 0.002256, 0.002259, 0.002261, 0.002262, 0.002261, 0.002259, 0.002256, 0.002251, 0.002244, 0.002237, 0.002228, 0.002217, 0.002205, 0.002192,
0.002179, 0.002192, 0.002204, 0.002214, 0.002223, 0.002231, 0.002237, 0.002242, 0.002246, 0.002248, 0.002248, 0.002248, 0.002246, 0.002242, 0.002237, 0.002231, 0.002223, 0.002214, 0.002204, 0.002192, 0.002179
};

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

__global__ void kernBlurUsingDiffs(int N, int width, int height, float * prevblur, uint8_t * prev, DiffPoint * diffs, int kernSize, float * blurKernel) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= N) {
    return;
  }
  DiffPoint & d = diffs[idx];
  for (int i = 0; i < kernSize; i++) {
    int tx = d.x + i - kernSize/2;
    for (int j = 0; j < kernSize; j++) {
      int ty = d.y + j - kernSize/2;
      if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
        int t = (ty * width + tx) * 3 + d.c;
        atomicAdd(&prevblur[t], (float)d.delta * blurKernel[j * kernSize + i]);
      }
    }
  }
	prev[(d.y * width + d.x) * 3 + d.c] = d.val;
  return;
}

__global__ void kernShowDiffs(int N, int width, int height, uint8_t * output, uint8_t * prev, DiffPoint * diffs) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= N) {
    return;
  }
  DiffPoint & d = diffs[idx];
	output[(d.y * width + d.x) * 3 + d.c] = d.delta;
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
	this->kernSize = 21;
	cudaMalloc(&this->dev_kernel, this->kernSize * this->kernSize * sizeof(float));
	cudaMemcpy(this->dev_kernel, kernel, this->kernSize * this->kernSize * sizeof(float), cudaMemcpyHostToDevice);
}

DiffBlur::~DiffBlur() {
  cudaFree(this->dev_diffPoints);
  cudaFree(this->dev_prev);
  cudaFree(this->dev_frame);
	cudaFree(this->dev_kernel);
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

	/*
  kernBlurUsingDiffs<<<blocksPerGrid, blockSize>>>(n_diffs, this->width, this->height, this->dev_prevblur, this->dev_prev, this->dev_diffPoints, this->kernSize, this->dev_kernel);
	kernCopyToFrame<<<blocksPerGrid, blockSize>>>(N, this->dev_frame, this->dev_prevblur);*/
	cudaMemset(this->dev_frame, 0, N * sizeof(uint8_t));
	kernShowDiffs<<<blocksPerGrid, blockSize>>>(n_diffs, this->width, this->height, this->dev_frame, this->dev_prev, this->dev_diffPoints);
  cudaMemcpy(frame, this->dev_frame, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  return 0;

}

AVPixelFormat DiffBlur::getPixelFormat() {
  return AV_PIX_FMT_RGB24;
}
