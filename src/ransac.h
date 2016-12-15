#pragma once

#include <bits/stdc++.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
using namespace std;

struct PointDelta {
  glm::vec2 delta;
  float dist;
  int origPos;
};

class RansacSeparator {
  private:
    const int BLOCK_SIZE = 128;
    int N;
    dim3 blocksPerGrid;

    bool * devPointGroup;
    glm::vec2 * devPointDiffs;
    PointDelta * devPointDeltas;
    thrust::device_ptr<PointDelta> thrust_devPointDeltas;
  public:
    RansacSeparator(int N);
    ~RansacSeparator();

    void separate(bool * pointGroup, glm::vec2 * pointDiffs, float THRESHOLD, int ITERATIONS);
};
