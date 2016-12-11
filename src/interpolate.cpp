#include <cstring>
#include <glm/vec2.hpp>
#include <algorithm>
#include <thrust/sort.h>
#include "interpolate.h"

#define NEIGHBORS 10
using namespace glm;
using namespace std;

typedef mot (pair<ivec2, ivec2>);

void knnInterpolate(int N, int width, int height, mot * sparse, mot * dense) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int * distance = new int[N];
      for (int k = 0; k < N; k++) {
        int dx = sparse[k].first.x - j;
        int dy = sparse[k].first.y - i;
        distance[k] = dx * dx + dy * dy;
      }
      thrust::sort(distance, distance + N, sparse);
      int totalDistance = 0;
      for (int k = 0; k < NEIGHBORS; k++) {
        totalDistance += distance[k];
      }
      float weightedX = 0;
      float weightedY = 0;
      for (int k = 0; k < NEIGHBORS; k++) {
        weightedX += (float) sparse[k].second.x * distance[k] / totalDistance;
        weightedY += (float) sparse[k].second.y * distance[k] / totalDistance;
      }
      dense[i * width + j] = pair<ivec2, ivec2>(ivec2(j,i), ivec2(weightedX, weightedY));
    }
  }
}

mot * interpolate(int N, int width, int height, mot * sparse) {
  mot * dense = new mot[N];
  knnInterpolate(N, width, height, sparse, dense);
}
