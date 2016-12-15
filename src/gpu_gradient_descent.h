#pragma once

#include <cuda.h>
#include <glm/glm.hpp>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

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
		int width;
		int height;
		int frames;
		glm::vec2 * devVO;
		glm::vec2 * devVB;
		float * devSequence;
		float * devImgO;
		float * devImgB;
		float * devAlpha;
		ImgData imgData;

		glm::vec2 * devVO_grad;
		glm::vec2 * devVB_grad;
		float * devImgO_grad;
		float * devImgB_grad;
		float * devAlpha_grad;
		GradData gradData;

		float * devW1;
		float * devW2;
		float * devW3;
		ImgWs imgWs;

		float * devObjective1;
		float * devObjective2;
		thrust::device_ptr<float> thrust_devObjective1;
		thrust::device_ptr<float> thrust_devObjective2;

		void optimizeImageComponents();
		void optimizeMotionFields();

  public:
		GpuGradientDescent(int width, int height, int frames, glm::vec2 * VO, glm::vec2 * VB, float * alpha, float * imgO, float * imgB, float * sequence);
    ~GpuGradientDescent();
    void optimize();
    float objectiveFunction();
		void getResults(ImgData * imgData);
};
