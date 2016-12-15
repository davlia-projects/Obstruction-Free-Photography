#pragma once

#include <glm/vec2.hpp>

class GradientDescent {
  private:
    const int GD_ROUNDS = 2;
    const int IMG_ROUNDS = 20;
    const float LEARNING_RATE1 = 1e-6;
    const int MOTION_ROUNDS = 20;
    const float LEARNING_RATE2 = 1e1;
    float LAMBDA_DT = 1.0f;
    const float LAMBDA_1 = 1.0f;
    const float LAMBDA_2 = 0.1f;
    const float LAMBDA_3 = 3000.0f;
    const float LAMBDA_4 = 0.5f;
    const float LAMBDA_P = 1e5;

    int width;
    int height;
    int frames;
    int frameCounter;
    // Grayscale sequence of frames
    float * sequence;

    void initializeBuffers();
    void optimizeImageComponents();
    void optimizeMotionFields();
    void freeBuffers();

    glm::vec2 * VB;
    glm::vec2 * VB_gd;
    glm::vec2 * VO;
    glm::vec2 * VO_gd;
    float * alpha;
    float * alpha_gd;
    float * imgO;
    float * imgO_gd;
    float * imgB;
    float * imgB_gd;

    float * w_1;
    float * w_2;
    float * w_3;

    int idx(int x, int y, int t = 0);
    int warpO(int x, int y, int t);
    int warpB(int x, int y, int t);
    glm::ivec2 iwarpO(int x, int y, int t);
    glm::ivec2 iwarpB(int x, int y, int t);
    glm::vec2 grad(float * field, int x, int y);
    glm::vec2 gradX(glm::vec2 * field, int x, int y, int t = 0);
    glm::vec2 gradY(glm::vec2 * field, int x, int y, int t = 0);
  public:
    GradientDescent(int width, int height, int frames, glm::vec2 * VB, glm::vec2 * VO, float * alpha, float * imgO, float * imgB, float * sequence);
    ~GradientDescent();
    void optimize();
    float objectiveFunction();
};
