#include "ransac.h"
#include "timing.h"
const int BLOCK_SIZE = 128;
static dim3 blocksPerGrid;

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
  blocksPerGrid = dim3((this->N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  cudaMalloc(&this->devPointDeltas, N * sizeof(PointDelta));
  cudaMalloc(&this->devPointDiffs, N * sizeof(glm::vec2));
  cudaMalloc(&this->devPointGroup, N * sizeof(bool));
  this->thrust_devPointDeltas = thrust::device_pointer_cast(this->devPointDeltas);
}

RansacSeparator::~RansacSeparator() {
  cudaFree(this->devPointDiffs);
  cudaFree(this->devPointDeltas);
  cudaFree(this->devPointGroup);
}

void RansacSeparator::computeDiffs(PointDelta & tempDelta, glm::vec2 & meanVector, int THRESHOLD_N, int ITERATIONS, float SCALE_THRESHOLD_N) {
  for (int i = 0; i < ITERATIONS; i++) {
    kernComputeDiffs<<<blocksPerGrid, BLOCK_SIZE>>>(this->N, meanVector, this->devPointDeltas);
    cudaDeviceSynchronize();
    thrust::sort(this->thrust_devPointDeltas, this->thrust_devPointDeltas + this->N, SortByDist());
    tempDelta.delta = glm::vec2(0.0f, 0.0f);
    tempDelta = thrust::reduce(this->thrust_devPointDeltas, this->thrust_devPointDeltas + THRESHOLD_N, tempDelta, AddDelta());
    meanVector = tempDelta.delta * SCALE_THRESHOLD_N;
  }
}

pair<glm::vec2,glm::vec2> RansacSeparator::separate(bool * pointGroup, glm::vec2 * pointDiffs, float THRESHOLD, int ITERATIONS) {
  cudaMemcpy(this->devPointDiffs, pointDiffs, this->N * sizeof(glm::vec2), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
  TIMEINIT
  TIMEIT((kernGeneratePointDeltas<<<blocksPerGrid, BLOCK_SIZE>>>(this->N, this->devPointDiffs, this->devPointDeltas)), "Generating Deltas")
	cudaDeviceSynchronize();

  int THRESHOLD_N = THRESHOLD * (float)this->N;
	printf("threshold: %d out of %d\n", THRESHOLD_N, this->N);
  float SCALE_THRESHOLD_N = 1.0f / (float)THRESHOLD_N;
	float SCALE_REMAINDER = 1.0f / (float) (this->N - THRESHOLD_N);

	PointDelta tempDelta;
  glm::vec2 meanVector(0.0f, 0.0f);
  TIMEIT(computeDiffs(tempDelta, meanVector, THRESHOLD_N, ITERATIONS, SCALE_THRESHOLD_N), "Computing Diffs")
	tempDelta.delta = glm::vec2(0.0f, 0.0f);
	tempDelta = thrust::reduce(this->thrust_devPointDeltas + THRESHOLD_N, this->thrust_devPointDeltas + this->N, tempDelta, AddDelta());
	tempDelta.delta *= SCALE_REMAINDER;
	printf("%f %f, %f %f\n", meanVector.x, meanVector.y, tempDelta.delta.x, tempDelta.delta.y);

	cudaMemset(this->devPointGroup, 0, this->N * sizeof(bool));
  TIMEIT((kernSetPointGroup<<<blocksPerGrid, BLOCK_SIZE>>>(THRESHOLD_N, this->devPointDeltas, this->devPointGroup)), "Set Point Group")
  TIMEEND
	cudaDeviceSynchronize();
  cudaMemcpy(pointGroup, this->devPointGroup, this->N * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
  return make_pair(meanVector, tempDelta.delta);
}
