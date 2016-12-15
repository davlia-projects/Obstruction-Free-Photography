#include "gpu_gradient_descent.h"
using namespace std;

const int GD_ROUNDS = 100;
const int IMG_ROUNDS = 200;
const int MOTION_ROUNDS = 20;
__device__ const float LEARNING_RATE1 = 1e-6;
__device__ const float LEARNING_RATE2 = 1e-4;
__device__ const float LAMBDA_DT = 1.0f;
__device__ const float LAMBDA_1 = 100.0f;
__device__ const float LAMBDA_2 = 0.1f;
__device__ const float LAMBDA_3 = 3000.0f;
__device__ const float LAMBDA_4 = 0.5f;
__device__ const float LAMBDA_P = 1e5;

__device__
int index(int x, int y, int width, int height) {
  return glm::clamp(y, 0, height - 1) * width + glm::clamp(x, 0, width - 1);
}

__device__
float phi(float t) {
  const float EPSILON_SQ = 1e-2;
  return sqrt(t * t + EPSILON_SQ);
}

__device__
int warpIdx(int width, int height, int x, int y, int z, glm::vec2 warp) {
  int nx = glm::clamp((int)(x + warp.x), 0, width - 1);
  int ny = glm::clamp((int)(y + warp.y), 0, height - 1);
  return z * width * height + ny * width + nx;
}

__device__
glm::vec2 grad(float * field, int x, int y, int width, int height) {
  float x1 = (x == 0) ? field[y * width + x] : field[y * width + (x - 1)];
  float x2 = (x == width - 1) ? field[y * width + x] : field[y * width + x + 1];
  float y1 = (y == 0) ? field[y * width + x] : field[(y - 1) * width + x];
  float y2 = (y == height - 1) ? field[y * width + x] : field[(y + 1) * width + x];
  return glm::vec2(x2 - x1, y2 - y1);
}

__global__
void kernImageGradientUpdate(int width, int height, ImgData imgData, GradData gradData) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  int idx = y * width + x;
  imgData.imgO[idx] -= LEARNING_RATE1 * gradData.imgO_grad[idx];
  imgData.imgB[idx] -= LEARNING_RATE1 * gradData.imgB_grad[idx];
  imgData.alpha[idx] -= LEARNING_RATE1 * gradData.alpha_grad[idx];

	imgData.imgO[idx] = glm::clamp(imgData.imgO[idx], -0.01f, 1.01f);
	imgData.imgB[idx] = glm::clamp(imgData.imgB[idx], -0.01f, 1.01f);
	imgData.alpha[idx] = glm::clamp(imgData.alpha[idx], -0.01f, 1.01f);
  return;
}

__global__
void kernComputeWTerms(int width, int height, int frames, ImgData imgData, ImgWs imgWs) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= frames) {
    return;
  }
  int idx = z * width * height + y * width + x;
  int warpO_idx = warpIdx(width, height, x, y, z, imgData.VO[idx]);
  int warpB_idx = warpIdx(width, height, x, y, z, imgData.VB[idx]);
  float dataTerm =
    LAMBDA_DT * (imgData.sequence[idx]
    - imgData.imgO[warpO_idx]
    - imgData.alpha[warpO_idx] * imgData.imgB[warpB_idx]);

  imgWs.W1[idx] = 1.0 / phi(dataTerm);
  glm::vec2 gradB = grad(imgData.imgB, x, y, width, height);
  imgWs.W2[idx] = 1.0f / phi(gradB.x * gradB.x + gradB.y * gradB.y);
  glm::vec2 gradO = grad(imgData.imgO, x, y, width, height);
  imgWs.W3[idx] = 1.0f / phi(gradO.x * gradO.x + gradO.y * gradO.y);
  return;
}

__global__
void kernComputeImageGradients1(int width, int height, int frames, ImgData imgData, GradData gradData, ImgWs imgWs) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= frames) {
    return;
  }
  int idx = y * width + x;
  int warpO_idx = warpIdx(width, height, x, y, z, imgData.VO[idx]);
  int warpB_idx = warpIdx(width, height, x, y, z, imgData.VB[idx]);
  float dataTerm =
    LAMBDA_DT * (imgData.sequence[idx]
    - imgData.imgO[warpO_idx]
    - imgData.alpha[warpO_idx] * imgData.imgB[warpB_idx]);

  gradData.imgO_grad[warpO_idx] -= dataTerm * imgWs.W1[idx];
  gradData.alpha_grad[warpO_idx] -= dataTerm * imgData.imgB[warpB_idx] * imgWs.W1[idx];
  gradData.imgB_grad[warpB_idx] -= dataTerm * imgData.alpha[warpO_idx] * imgWs.W1[idx];
  return;
}

__global__
void kernComputeImageGradients2(int width, int height, ImgData imgData, GradData gradData, ImgWs imgWs) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  int idx = y * width + x;
	glm::vec2 g;
	g = grad(imgData.alpha, x + 1, y, width, height);
  gradData.alpha_grad[idx] += g.x;
	g = grad(imgData.alpha, x - 1, y, width, height);
  gradData.alpha_grad[idx] += g.x;
	g = grad(imgData.alpha, x, y + 1, width, height);
  gradData.alpha_grad[idx] += g.y;
	g = grad(imgData.alpha, x, y - 1, width, height);
  gradData.alpha_grad[idx] += g.y;

  g = grad(imgData.imgO, x + 1, y, width, height);
  gradData.imgO_grad[idx] -= LAMBDA_2 * imgWs.W2[index(x + 1, y, width, height)] * g.x;
  gradData.imgO_grad[idx] += LAMBDA_3 * (g.x * g.x + g.y * g.y) * g.x;
  g = grad(imgData.imgO, x - 1, y, width, height);
  gradData.imgO_grad[idx] += LAMBDA_2 * imgWs.W2[index(x - 1, y, width, height)] * g.x;
  gradData.imgO_grad[idx] += LAMBDA_3 * (g.x * g.x + g.y * g.y) * g.x;
  g = grad(imgData.imgO, x, y + 1, width, height);
  gradData.imgO_grad[idx] -= LAMBDA_2 * imgWs.W2[index(x, y + 1, width, height)] * g.y;
  gradData.imgO_grad[idx] += LAMBDA_3 * (g.x * g.x + g.y * g.y) * g.y;
  g = grad(imgData.imgO, x, y - 1, width, height);
  gradData.imgO_grad[idx] += LAMBDA_2 * imgWs.W2[index(x, y - 1, width, height)] * g.y;
  gradData.imgO_grad[idx] += LAMBDA_3 * (g.x * g.x + g.y * g.y) * g.y;

  g = grad(imgData.imgB, x + 1, y, width, height);
  gradData.imgB_grad[idx] -= LAMBDA_2 * imgWs.W3[index(x + 1, y, width, height)] * g.x;
  g = grad(imgData.imgB, x - 1, y, width, height);
  gradData.imgB_grad[idx] += LAMBDA_2 * imgWs.W3[index(x - 1, y, width, height)] * g.x;
  g = grad(imgData.imgB, x, y + 1, width, height);
  gradData.imgB_grad[idx] -= LAMBDA_2 * imgWs.W3[index(x, y + 1, width, height)] * g.y;
  g = grad(imgData.imgB, x, y - 1, width, height);
  gradData.imgB_grad[idx] += LAMBDA_2 * imgWs.W3[index(x, y - 1, width, height)] * g.y;

  if (imgData.imgO[idx] < 0.0f) {
    gradData.imgO_grad[idx] += LAMBDA_P * imgData.imgO[idx];
  }
  if (imgData.imgB[idx] < 0.0f) {
    gradData.imgB_grad[idx] += LAMBDA_P * imgData.imgB[idx];
  }
  if (imgData.alpha[idx] < 0.0f) {
    gradData.alpha_grad[idx] += LAMBDA_P * imgData.alpha[idx];
  }

  if (imgData.imgO[idx] > 1.0f) {
    gradData.imgO_grad[idx] -= LAMBDA_P * (1.0f - imgData.imgO[idx]);
  }
  if (imgData.imgB[idx] > 1.0f) {
    gradData.imgB_grad[idx] -= LAMBDA_P * (1.0f - imgData.imgB[idx]);
  }
  if (imgData.alpha[idx] > 1.0f) {
    gradData.alpha_grad[idx] -= LAMBDA_P * (1.0f - imgData.alpha[idx]);
  }
  return;
}

__global__
void kernComputeMotionGradients(int width, int height, int frames, ImgData imgData, GradData gradData) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= frames) {
    return;
  }
	int idx = z * width * height + y * width + x;
	int idx2 = y * width + x;
  int warpO_idx = warpIdx(width, height, x, y, z, imgData.VO[idx]);
  int warpB_idx = warpIdx(width, height, x, y, z, imgData.VB[idx]);

  float dataTerm =
    LAMBDA_DT * (imgData.sequence[idx]
    - imgData.imgO[warpO_idx]
    - imgData.alpha[warpO_idx] * imgData.imgB[warpB_idx]);

	glm::ivec2 wo;
	glm::vec2 motion = imgData.VO[idx];
	glm::vec2 g;
	wo.x = glm::clamp(x + (int)motion.x, 0, width - 1);
	wo.y = glm::clamp(y + (int)motion.y, 0, height - 1);

	g = grad(imgData.imgO, wo.x, wo.y, width, height);
	gradData.VO_grad[idx] -= dataTerm * g;
	g = grad(imgData.imgB, wo.x, wo.y, width, height);
	gradData.VB_grad[idx] -= dataTerm * imgData.alpha[warpO_idx] * g;
	gradData.VO_grad[idx] -= dataTerm * imgData.imgB[idx2] * grad(imgData.alpha, wo.x, wo.y, width, height);
}

__global__
void kernMotionGradientUpdate(int width, int height, int frames, ImgData imgData, GradData gradData) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= frames) {
    return;
  }
  int idx = z * width * height + y * width + x;
	imgData.VO[idx] -= LEARNING_RATE2 * gradData.VO_grad[idx];
	imgData.VB[idx] -= LEARNING_RATE2 * gradData.VB_grad[idx];
}

__global__
void kernObjectiveDataTerm(int width, int height, int frames, ImgData imgData, float * objective) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height || z >= frames) {
    return;
  }
  int idx = z * width * height + y * width + x;
  int warpO_idx = warpIdx(width, height, x, y, z, imgData.VO[idx]);
  int warpB_idx = warpIdx(width, height, x, y, z, imgData.VB[idx]);
  float pixRet = 0.0f;
  pixRet += imgData.sequence[idx];
  pixRet -= imgData.imgO[warpO_idx];
  pixRet -= imgData.alpha[warpO_idx] * imgData.imgB[warpB_idx];

  objective[idx] = glm::abs(pixRet) * LAMBDA_DT;
  return;
}

__global__
void kernObjectiveOtherTerms(int width, int height, ImgData imgData, float * objective) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  float ret = 0.0f;
  glm::vec2 alpha_grad = grad(imgData.alpha, x, y, width, height);
  ret += LAMBDA_1 * (alpha_grad.x * alpha_grad.x + alpha_grad.y * alpha_grad.y);
  glm::vec2 imgO_grad = grad(imgData.imgO, x, y, width, height);
  ret += LAMBDA_2 * (glm::abs(imgO_grad.x) + glm::abs(imgO_grad.y));
  glm::vec2 imgB_grad = grad(imgData.imgB, x, y, width, height);
  ret += LAMBDA_2 * (glm::abs(imgB_grad.x) + glm::abs(imgB_grad.y));
  ret += LAMBDA_3 
    * (imgO_grad.x * imgO_grad.x + imgO_grad.y * imgO_grad.y)
    * (imgB_grad.x * imgB_grad.x + imgB_grad.y * imgB_grad.y);
  // TODO: too lazy to do sparsity constraint 

  objective[y * width + x] = ret;
  return;
}


GpuGradientDescent::GpuGradientDescent(int width, int height, int frames, glm::vec2 * VO, glm::vec2 * VB, float * alpha, float * imgO, float * imgB, float * sequence) {
  int N = width * height;
  int N2 = width * height * frames;
  this->width = width;
  this->height = height;
  this->frames = frames;
  cudaMalloc(&this->devVO, N2 * sizeof(glm::vec2));
  cudaMemcpy(this->devVO, VO, N2 * sizeof(glm::vec2), cudaMemcpyHostToDevice);
  cudaMalloc(&this->devVB, N2 * sizeof(glm::vec2));
  cudaMemcpy(this->devVB, VB, N2 * sizeof(glm::vec2), cudaMemcpyHostToDevice);
  cudaMalloc(&this->devSequence, N * sizeof(float));
  cudaMemcpy(this->devSequence, sequence, N2 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&this->devImgO, N * sizeof(float));
  cudaMemcpy(this->devImgO, imgO, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&this->devImgB, N * sizeof(float));
  cudaMemcpy(this->devImgB, imgB, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&this->devAlpha, N * sizeof(float));
  cudaMemcpy(this->devAlpha, alpha, N * sizeof(float), cudaMemcpyHostToDevice);
  this->imgData.VO = this->devVO;
  this->imgData.VB = this->devVB;
  this->imgData.alpha = this->devAlpha;
  this->imgData.imgO = this->devImgO;
  this->imgData.imgB = this->devImgB;
  this->imgData.sequence = this->devSequence;

  cudaMalloc(&this->devVO_grad, N2 * sizeof(glm::vec2));
  cudaMalloc(&this->devVB_grad, N2 * sizeof(glm::vec2));
  cudaMalloc(&this->devImgO_grad, N * sizeof(float));
  cudaMalloc(&this->devImgB_grad, N * sizeof(float));
  cudaMalloc(&this->devAlpha_grad, N * sizeof(float));
  this->gradData.VO_grad = this->devVO_grad;
  this->gradData.VB_grad = this->devVB_grad;
  this->gradData.alpha_grad = this->devAlpha_grad;
  this->gradData.imgO_grad = this->devImgO_grad;
  this->gradData.imgB_grad = this->devImgB_grad;

  cudaMalloc(&this->devW1, N2 * sizeof(float));
  cudaMalloc(&this->devW2, N2 * sizeof(float));
  cudaMalloc(&this->devW3, N2 * sizeof(float));
  this->imgWs.W1 = this->devW1;
  this->imgWs.W2 = this->devW2;
  this->imgWs.W3 = this->devW3;

	cudaMalloc(&this->devObjective1, N2 * sizeof(float));
	cudaMalloc(&this->devObjective2, N * sizeof(float));
	this->thrust_devObjective1 = thrust::device_pointer_cast(this->devObjective1);
	this->thrust_devObjective2 = thrust::device_pointer_cast(this->devObjective2);
}

GpuGradientDescent::~GpuGradientDescent() {
  cudaFree(this->devVO);
  cudaFree(this->devVB);
  cudaFree(this->devImgO);
  cudaFree(this->devImgB);
  cudaFree(this->devAlpha);
  cudaFree(this->devVO_grad);
  cudaFree(this->devVB_grad);
  cudaFree(this->devImgO_grad);
  cudaFree(this->devImgB_grad);
  cudaFree(this->devAlpha_grad);
  cudaFree(this->devW1);
  cudaFree(this->devW2);
  cudaFree(this->devW3);
}

void GpuGradientDescent::optimize() {
	for (int i = 0; i < GD_ROUNDS; i++) {
		for (int j = 0; j < MOTION_ROUNDS; j++) {
			this->optimizeMotionFields();
		}
    for (int j = 0; j < IMG_ROUNDS; j++) {
      this->optimizeImageComponents();
    }
		if (i % 100 == 0) {
      printf("OBJECTIVE: %f\n", this->objectiveFunction());
		}
  }
}

void GpuGradientDescent::optimizeImageComponents() {
  int N = this->width * this->height;
  // Initialize gradients to 0
  cudaMemset(this->devImgO_grad, 0, N * sizeof(float));
  cudaMemset(this->devImgB_grad, 0, N * sizeof(float));
  cudaMemset(this->devAlpha_grad, 0, N * sizeof(float));

  dim3 blockSize2d(16, 16);
  dim3 blocksPerGrid2d(
    (this->width + blockSize2d.x - 1) / blockSize2d.x,
    (this->height + blockSize2d.y - 1) / blockSize2d.y);
  dim3 blockSize3d(8, 8, 5);
  dim3 blocksPerGrid3d(
    (this->width + blockSize3d.x - 1) / blockSize3d.x,
    (this->height + blockSize3d.y - 1) / blockSize3d.y,
    (this->frames + blockSize3d.z - 1) / blockSize3d.z);

  kernComputeWTerms<<<blocksPerGrid3d, blockSize3d>>>(
      this->width, this->height, this->frames, this->imgData, this->imgWs);

  kernComputeImageGradients1<<<blocksPerGrid3d, blockSize3d>>>(
      this->width, this->height, this->frames, this->imgData, this->gradData, this->imgWs);
  kernComputeImageGradients2<<<blocksPerGrid2d, blockSize2d>>>(
      this->width, this->height, this->imgData, this->gradData, this->imgWs);

  kernImageGradientUpdate<<<blocksPerGrid2d, blockSize2d>>>(
      this->width, this->height, this->imgData, this->gradData);
}

void GpuGradientDescent::optimizeMotionFields() {
	int N = this->width * this->height * this->frames;
	cudaMemset(this->devVO_grad, 0, N * sizeof(glm::vec2));
	cudaMemset(this->devVB_grad, 0, N * sizeof(glm::vec2));

	dim3 blockSize2d(16, 16);
  dim3 blocksPerGrid2d(
    (this->width + blockSize2d.x - 1) / blockSize2d.x,
    (this->height + blockSize2d.y - 1) / blockSize2d.y);
  dim3 blockSize3d(8, 8, 5);
  dim3 blocksPerGrid3d(
    (this->width + blockSize3d.x - 1) / blockSize3d.x,
    (this->height + blockSize3d.y - 1) / blockSize3d.y,
    (this->frames + blockSize3d.z - 1) / blockSize3d.z);

	kernComputeMotionGradients<<<blocksPerGrid3d, blockSize3d>>>(
		this->width, this->height, this->frames, this->imgData, this->gradData);

	kernMotionGradientUpdate<<<blocksPerGrid3d, blockSize3d>>>(
		this->width, this->height, this->frames, this->imgData, this->gradData);
}

void GpuGradientDescent::getResults(ImgData * imgData) {
  int N = this->width * this->height;
  int N2 = N * this->frames;
  cudaMemcpy(imgData->VO, this->imgData.VO, N2 * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
  cudaMemcpy(imgData->VB, this->imgData.VB, N2 * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
  cudaMemcpy(imgData->alpha, this->imgData.alpha, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(imgData->imgO, this->imgData.imgO, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(imgData->imgB, this->imgData.imgB, N * sizeof(float), cudaMemcpyDeviceToHost);
  return;
}

float GpuGradientDescent::objectiveFunction() {
  float ret = 0.0f;
  dim3 blockSize2d(16, 16);
  dim3 blocksPerGrid2d(
    (this->width + blockSize2d.x - 1) / blockSize2d.x,
    (this->height + blockSize2d.y - 1) / blockSize2d.y);
  dim3 blockSize3d(8, 8, 5);
  dim3 blocksPerGrid3d(
    (this->width + blockSize3d.x - 1) / blockSize3d.x,
    (this->height + blockSize3d.y - 1) / blockSize3d.y,
    (this->frames + blockSize3d.z - 1) / blockSize3d.z);

  kernObjectiveDataTerm<<<blocksPerGrid3d, blockSize3d>>>(
      this->width, this->height, this->frames, this->imgData, this->devObjective1);
  kernObjectiveOtherTerms<<<blocksPerGrid2d, blockSize2d>>>(
      this->width, this->height, this->imgData, this->devObjective2);

  int N = this->width * this->height;
  int N2 = N * this->frames;

  ret += thrust::reduce(this->thrust_devObjective1, this->thrust_devObjective1 + N2, 0.0f);
  ret += thrust::reduce(this->thrust_devObjective2, this->thrust_devObjective2 + N, 0.0f);
  return ret;
}
