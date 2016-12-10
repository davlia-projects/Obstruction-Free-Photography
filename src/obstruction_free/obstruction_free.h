#pragma once

#include <glm/vec2.hpp>
#include "pipeline.h"

class ObstructionFree: public Pipeline {
  private:
    const int GD_ROUNDS = 3;
    const int IMG_ROUNDS = 5;
    const float LEARNING_RATE1 = 1e-3;
    const int MOTION_ROUNDS = 10;
    const float LEARNING_RATE2 = 1e3;
    const float LAMBDA_1 = 1.0f;
    const float LAMBDA_2 = 1.0f;
    const float LAMBDA_3 = 1.0f;
    const float LAMBDA_4 = 1.0f;
    const float LAMBDA_P = 1e5;

    int width;
    int height;
    int frames;
    int frameCounter;
    // Grayscale sequence of frames
    float * sequence;

    void removeObstructions();
    void initializeBuffers();
    void optimizeImageComponents();
    void optimizeMotionFields();
    void freeBuffers();

    float objectiveFunction();
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
    ObstructionFree(int width, int height, int frames);
    ~ObstructionFree();
    int processFrame(uint8_t * frame);
    AVPixelFormat getPixelFormat();
};
