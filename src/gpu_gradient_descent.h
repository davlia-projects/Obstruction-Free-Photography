#pragma once

#include <cuda.h>

struct ImgData {
  glm::vec2 * VO;
  glm::vec2 * VB;
  float * alpha;
  float * imgO;
  float * imgB;
  float * sequence;
};

struct GradData {
  glm::vec2 * VO_grad;
  glm::vec2 * VB_grad;
  float * alpha_grad;
  float * imgO_grad;
  float * imgB_grad;
};

struct ImgWs {
  float * W1;
  float * W2;
  float * W3;
};

class GpuGradientDescent {
  private:
  public:
    GpuGradientDescent(int width, int height, int frames, float * sequence);
    ~GpuGradientDescent();
    void optimize();
    float objectiveFunction();
};
