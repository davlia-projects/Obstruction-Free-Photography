#include "canny.h"
#include <cstdio>
#include <cmath>

#define UPPERTHRESHOLD 90
#define LOWERTHRESHOLD 30

const float G_x[3 * 3] = {
  -1, 0, 1,
  -2, 0, 2,
  -1, 0, 1
};

const float G_y[3 * 3] = {
  1, 2, 1,
  0, 0, 0,
  -1, -2, -1
};

const float gaussian[5 * 5] = {
  2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159,
  4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
  5.f/159, 12.f/159, 15.f/159, 12.f/159, 2.f/159,
  4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
  2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159
};


__global__ void kernSmooth(int N, int width, int height, unsigned char * in, unsigned char * out, const float * kernel, int kernSize) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}
  float c = 0.0f;
	for (int i = 0; i < kernSize; i++) {
		int tx = x + i - kernSize/2;
		for (int j = 0; j < kernSize; j++) {
			int ty = y + j - kernSize/2;
			if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
				c += in[ty * width + tx] * kernel[j * kernSize + i];
			}
		}
	}
	out[y * width + x] = fabs(c);
}

__global__ void kernGradient(int N, int width, int height, unsigned char * in, unsigned char * gradient, unsigned char * edgeDir, unsigned char * G_x, unsigned char * G_y) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}
  int idx, dx, dy, tx, ty;
  float Gx, Gy, grad, angle;
  idx = y * width + x;
  Gx = Gy = 0;
  for (dy = 0; dy < 3; dy++) {
    ty = y + dy - 1;
    for (dx = 0; dx < 3; dx++) {
      tx = x + dx - 1;
      if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
        Gx += in[ty * width + tx] * G_x[dy * 3 + dx];
        Gy += in[ty * width + tx] * G_y[dy * 3 + dx];
      }
    }
  }
  grad = sqrt(Gx * Gx + Gy * Gy);
  angle = (atan2(Gx, Gy) / 3.14159f) * 180.0f;
  unsigned char roundedAngle;
  if (((-22.5 < angle) && (angle <= 22.5)) || ((157.5 < angle) && (angle <= -157.5))) {
    roundedAngle = 0;
  }
	if (((-157.5 < angle) && (angle <= -112.5)) || ((22.5 < angle) && (angle <= 67.5))) {
    roundedAngle = 45;
  }
	if (((-112.5 < angle) && (angle <= -67.5)) || ((67.5 < angle) && (angle <= 112.5))) {
    roundedAngle = 90;
  }
	if (((-67.5 < angle) && (angle <= -22.5)) || ((112.5 < angle) && (angle <= 157.5))) {
    roundedAngle = 135;
  }
  gradient[idx] = grad;
  edgeDir[idx] = roundedAngle;
}



__global__ void nonMaxSuppression(int N, int width, int height, unsigned char * in, unsigned char * out) {
  int D = 1;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}
  int angle = in[y * width + x];
  switch(angle) {
    case 0:
      if (out[y * width + x] < out[(y + D) * width + x] || out[y * width + x] < out[(y - D) * width + x]) {
        out[y * width + x] = 0;
      }
      break;
    case 45:
      if (out[y * width + x] < out[(y + D) * width + x - D] || out[y * width + x] < out[(y - D) * width + x + D]) {
        out[y * width + x] = 0;
      }
      break;
    case 90:
      if (out[y * width + x] < out[y * width + x + D] || out[y * width + x] < out[y * width + x - D]) {
        out[y * width + x] = 0;
      }
      break;

    case 135:
      if (out[y * width + x] < out[(y + D) * width + x + D] || out[y * width + x] < out[(y - D) * width + x - D]) {
        out[y * width + x] = 0;
      }
      break;
    default:
      break;
  }
}

__global__ void hysteresis(int N, int width, int height, unsigned char * in) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}
  int idx = y * width + x;
  if (in[idx] > UPPERTHRESHOLD) {
    in[idx] = 255;
  } else if (in[idx] < LOWERTHRESHOLD) {
    in[idx] = 0;
  } else {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int nidx = (y + dy) * width + (x + dx);
        if(0 <= (y + dy) && (y + dy) < height &&  0 <= (x + dx) && (x + dx) < width && in[nidx] > LOWERTHRESHOLD) {
          in[nidx] = 255;
        }
      }
    }
  }
}


unsigned char * Canny::edge(int N, int width, int height, unsigned char * in) {

  unsigned char * smooth, * gradient, * edgeDir, * gradient_x, * gradient_y;
  float * blur;
  cudaMalloc(&smooth, sizeof(unsigned char) * N);
  cudaMalloc(&gradient, sizeof(unsigned char) * N);
  cudaMalloc(&edgeDir, sizeof(unsigned char) * N);

  cudaMalloc(&blur, sizeof(float) * 5 * 5);
  cudaMemcpy(blur, gaussian, sizeof(float) * 5 * 5, cudaMemcpyHostToDevice);

  cudaMalloc(&gradient_x, sizeof(unsigned char) * 3 * 3);
  cudaMemcpy(gradient_x, G_x, sizeof(unsigned char) * 3 * 3, cudaMemcpyHostToDevice);

  cudaMalloc(&gradient_y, sizeof(unsigned char) * 3 * 3);
  cudaMemcpy(gradient_y, G_y, sizeof(unsigned char) * 3 * 3, cudaMemcpyHostToDevice);

  const dim3 blockSize2d(8,8);
  const dim3 blocksPerGrid2d(
  		(width + blockSize2d.x - 1) / blockSize2d.x,
  		(height + blockSize2d.y - 1) / blockSize2d.y);

  kernSmooth<<<blocksPerGrid2d, blockSize2d>>>(N, width, height, in, smooth, blur, 5);
  kernGradient<<<blocksPerGrid2d, blockSize2d>>>(N, width, height, smooth, gradient, edgeDir, gradient_x, gradient_y);
  nonMaxSuppression<<<blocksPerGrid2d, blockSize2d>>>(N, width, height, edgeDir, gradient);
  hysteresis<<<blocksPerGrid2d, blockSize2d>>>(N, width, height, gradient); // can use stream compaction

  cudaFree(smooth);
  cudaFree(edgeDir);
  cudaFree(blur);
  cudaFree(gradient_x);
  cudaFree(gradient_y);

  return gradient;
}
