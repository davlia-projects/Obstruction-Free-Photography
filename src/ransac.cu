#include "ransac.h"

__global__
void kernComputeDiffs(int N, glm::vec2 v, PointDelta * pointDeltas) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  pointDeltas[idx].dist = glm::distance(v, pointDeltas[idx].delta);
  return;
}

__global__
void kernGeneratePointDeltas(int N, glm::vec2 * pointDiffs, PointDelta * pointDeltas) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  pointDeltas[idx].origPos = idx;
  pointDeltas[idx].delta = pointDiffs[idx];
  return;
}

__global__
void kernSetPointGroup(int N, PointDelta * pointDeltas, bool * pointGroup) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  pointGroup[pointDeltas[idx].origPos] = true;
  return;
}

struct SortByDist {
  __host__ __device__
  bool operator()(const PointDelta & d1, const PointDelta & d2) {
    return d1.dist < d2.dist;
  }
};

struct AddDelta {
  __host__ __device__
  PointDelta operator()(const PointDelta & d1, const PointDelta & d2) {
    PointDelta pd;
    pd.delta = d1.delta + d2.delta;
    return pd;
  }
};

RansacSeparator::RansacSeparator(int N) {
  this->N = N;
  this->blocksPerGrid = dim3((this->N + this->BLOCK_SIZE - 1) / this->BLOCK_SIZE);
  cudaMalloc(&this->devPointDeltas, N * sizeof(PointDelta));
  cudaMalloc(&this->devPointDiffs, N * sizeof(glm::vec2));
  this->thrust_devPointDeltas = thrust::device_pointer_cast(this->devPointDeltas);
}

RansacSeparator::~RansacSeparator() {
  cudaFree(this->devPointDiffs);
  cudaFree(this->devPointDeltas);
  cudaFree(this->devPointGroup);
}

glm::vec2 RansacSeparator::separate(bool * pointGroup, glm::vec2 * pointDiffs, float THRESHOLD, float ITERATIONS) {
  cudaMemcpy(this->devPointDiffs, pointDiffs, this->N * sizeof(glm::vec2), cudaMemcpyHostToDevice);
  kernGeneratePointDeltas<<<this->blocksPerGrid, this->BLOCK_SIZE>>>(this->N, this->devPointDiffs, this->devPointDeltas);

  int THRESHOLD_N = THRESHOLD * this->N;
  float SCALE_THRESHOLD_N = 1.0f / (float)THRESHOLD_N;

  glm::vec2 meanVector(0.0f, 0.0f);
  for (int i = 0; i < ITERATIONS; i++) {
    kernComputeDiffs<<<this->blocksPerGrid, this->BLOCK_SIZE>>>(this->N, meanVector, this->devPointDiffs, this->devDists);
    thrust::sort(this->thrust_devPointDeltas, this->thrust_devPointDeltas + this->N, SortByDist());
    PointDelta tempDelta;
    tempDelta.delta = glm::vec2(0.0f, 0.0f);
    thrust::reduce(this->thrust_devPointdeltas, this->thrust_devPointDeltas + THRESHOLD_N, tempDelta, AddDelta());
    meanVector = tempDelta.delta * SCALE_THRESHOLD_N;
  }

  cudaMemcpy(pointGroup, this->devPointGroup, this->N * sizeof(bool), cudaMemcpyDeviceToHost);
  return meanVector;
}
