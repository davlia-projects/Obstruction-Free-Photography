#pragma once

#include <cuda.h>
#include <glm/glm.hpp>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <utility>
using namespace std;

struct PointDelta {
  glm::vec2 delta;
  float dist;
  int origPos;
};

class RansacSeparator {
  private:
    bool * devPointGroup;
    glm::vec2 * devPointDiffs;
    PointDelta * devPointDeltas;
    thrust::device_ptr<PointDelta> thrust_devPointDeltas;
  public:
		int N;
    RansacSeparator(int N);
    ~RansacSeparator();

    pair<glm::vec2, glm::vec2> separate(bool * pointGroup, glm::vec2 * pointDiffs, float THRESHOLD, int ITERATIONS);
};
