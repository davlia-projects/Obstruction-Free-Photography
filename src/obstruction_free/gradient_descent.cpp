// #include <cuda.h>
#include <glm/vec2.hpp>
#include <bits/stdc++.h>
using namespace std;

static int WIDTH = 640;
static int HEIGHT = 480;

float * w_1;
float * w_2;
float * w_3;
float * img_obstruction_grad;
float * img_background_grad;
float * alpha_grad;
glm::vec2 * motion_obstruction_grad;
glm::vec2 * motion_background_grad;
glm::vec2 * img_obstruction_deriv;
glm::vec2 * img_background_deriv;
glm::vec2 * alpha_deriv;
float * img_obstruction_deriv_sq;
float * img_background_deriv_sq;
float * alpha_deriv_sq;
float * motion_obstruction_deriv_sq;
float * motion_background_deriv_sq;

void initializeBuffers(int width, int height) {
  WIDTH = width;
  HEIGHT = height;
  int sz = width * height;
  w_1 = (float *) malloc(sz * sizeof(float));
  w_2 = (float *) malloc(sz * sizeof(float));
  w_3 = (float *) malloc(sz * sizeof(float));
  img_obstruction_grad = (float *) malloc(sz * sizeof(float));
  img_background_grad = (float *) malloc(sz * sizeof(float));
  alpha_grad = (float *) malloc(sz * sizeof(float));
  img_obstruction_deriv_sq = (float *) malloc(sz * sizeof(float));
  img_background_deriv_sq = (float *) malloc(sz * sizeof(float));
  alpha_deriv_sq = (float *) malloc(sz * sizeof(float));
  motion_obstruction_deriv_sq = (float *) malloc(sz * sizeof(float));
  motion_background_deriv_sq = (float *) malloc(sz * sizeof(float));

  motion_obstruction_grad = (glm::vec2 *) malloc(sz * sizeof(glm::vec2));
  motion_background_grad = (glm::vec2 *) malloc(sz * sizeof(glm::vec2));
  img_obstruction_deriv = (glm::vec2 *) malloc(sz * sizeof(glm::vec2));
  img_background_deriv = (glm::vec2 *) malloc(sz * sizeof(glm::vec2));
  alpha_deriv = (glm::vec2 *) malloc(sz * sizeof(glm::vec2));
  return;
}

float objectiveFunction(float * img, float * img_obstruction, float * img_background, float * alpha, glm::vec2 * motion_obstruction, glm::vec2 * motion_background) {
  float ret = 0.0f;
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      ret += fabs(img[idx(x, y)] - img_obstruction[warp(motion_obstruction, x, y)] - alpha[warp(motion_obstruction, x, y)] * img_background[warp(motion_background, x, y)]);
      ret += LAMBDA_1 * alpha_deriv_sq[idx(x, y)];
      ret += LAMBDA_2 * (fabs(img_obstruction_deriv[idx(x, y)].x) + fabs(img_obstruction_deriv[idx(x, y)].y));
      ret += LAMBDA_2 * (fabs(img_background_deriv[idx(x, y)].x) + fabs(img_background_deriv[idx(x, y)].y));
      ret += LAMBDA_3 * img_obstruction_deriv_sq[idx(x, y)] * img_background_deriv_sq[idx(x, y)];
      ret += LAMBDA_4 * (fabs(motion_obstruction_deriv[idx(x, y)].x) + fabs(motion_obstruction_deriv[idx(x, y)].y));
      ret += LAMBDA_4 * (fabs(motion_background_deriv[idx(x, y)].x) + fabs(motion_background_deriv[idx(x, y)].y));
    }
  }
  return ret;
}

float phi(float x) {
  const float EPSILON_SQ = 1e-10;
  return sqrt(x * x + EPSILON_SQ);
}

int idx(int x, int y) {
  x = max(0, min(WIDTH - 1, x));
  y = max(0, min(HEIGHT - 1, y));
  return y * WIDTH + x;
}

int warp(glm::vec2 * motion, int x, int y) {
  glm::vec2 v = motion[idx(x, y)];
  return idx(x + v.x, y + v.y);
}

void computeDerivatives(float * img, float * img_obstruction, float * img_background, float * alpha, glm::vec2 * motion_obstruction, glm::vec2 * motion_background) {
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      if (x == 0) {
        img_obstruction_deriv[idx(x, y)].x = 2.0f * (img_obstruction[idx(x + 1, y)] - img_obstruction[idx(x, y)]);
        img_background_deriv[idx(x, y)].x = 2.0f * (img_background[idx(x + 1, y)] - img_background[idx(x, y)]);
        alpha_deriv[idx(x, y)].x = 2.0f * (alpha[idx(x + 1, y)] - alpha[idx(x, y)]);
      } else if (x == WIDTH - 1) {
        img_obstruction_deriv[idx(x, y)].x = 2.0f * (img_obstruction[idx(x, y)] - img_obstruction[idx(x - 1, y)]);
        img_background_deriv[idx(x, y)].x = 2.0f * (img_background[idx(x, y)] - img_background[idx(x - 1, y)]);
        alpha_deriv[idx(x, y)].x = 2.0f * (alpha[idx(x, y)] - alpha[idx(x - 1, y)]);
      } else {
        img_obstruction_deriv[idx(x, y)].x = img_obstruction[idx(x + 1, y)] - img_obstruction[idx(x - 1, y)];
        img_background_deriv[idx(x, y)].x = img_background[idx(x + 1, y)] - img_background[idx(x - 1, y)];
        alpha_deriv[idx(x, y)].x = alpha[idx(x + 1, y)] - alpha[idx(x - 1, y)];
      }
      if (y == 0) {
        img_obstruction_deriv[idx(x, y)].y = 2.0f * (img_obstruction[idx(x, y)] - img_obstruction[idx(x, y + 1)]);
        img_background_deriv[idx(x, y)].y = 2.0f * (img_background[idx(x, y)] - img_background[idx(x, y + 1)]);
        alpha_deriv[idx(x, y)].y = 2.0f * (alpha[idx(x, y)] - alpha[idx(x, y + 1)]);
      } else if (y == HEIGHT - 1) {
        img_obstruction_deriv[idx(x, y)].y = 2.0f * (img_obstruction[idx(x, y)] - img_obstruction[idx(x, y - 1)]);
        img_background_deriv[idx(x, y)].y = 2.0f * (img_background[idx(x, y)] - img_background[idx(x, y - 1)]);
        alpha_deriv[idx(x, y)].y = 2.0f * (alpha[idx(x, y)] - alpha[idx(x, y - 1)]);
      } else {
        img_obstruction_deriv[idx(x, y)].y = img_obstruction[idx(x, y + 1)] - img_obstruction[idx(x, y - 1)];
        img_background_deriv[idx(x, y)].y = img_background[idx(x, y + 1)] - img_background[idx(x, y - 1)];
        alpha_deriv[idx(x, y)].y = alpha[idx(x, y + 1)] - alpha[idx(x, y - 1)];
      }
      img_obstruction_deriv_sq[idx(x, y)] = img_obstruction_deriv[idx(x, y)].x * img_obstruction_deriv[idx(x, y)].x + img_obstruction_deriv[idx(x, y)].y * img_obstruction_deriv[idx(x, y)].y;
      img_background_deriv_sq[idx(x, y)] = img_background_deriv[idx(x, y)].x * img_background_deriv[idx(x, y)].x + img_background_deriv[idx(x, y)].y * img_background_deriv[idx(x, y)].y;
    }
  }
  return;
}

void computeWs1(float * img, float * img_obstruction, float * img_background, float * alpha, glm::vec2 * motion_obstruction, glm::vec2 * motion_background) {
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      float t;
      t = img[idx(x, y)] - img_obstruction[warp(motion_obstruction, x, y)] - alpha[warp(motion_obstruction, x, y)] * img_background[warp(motion_background, x, y)];
      w_1[idx(x, y)] = 1.0f / phi(t);
      w_2[idx(x, y)] = 1.0f / phi(img_background_deriv_sq[idx(x, y)]);
      w_3[idx(x, y)] = 1.0f / phi(img_obstruction_deriv_sq[idx(x, y)]);
    }
  }
}


void computeWs2(float * img, float * img_obstruction, float * img_background, float * alpha, glm::vec2 * motion_obstruction, glm::vec2 * motion_background) {
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      float t;
      t = img[idx(x, y)] - img_obstruction[warp(motion_obstruction, x, y)] - alpha[warp(motion_obstruction, x, y)] * img_background[warp(motion_background, x, y)];
      w_1[idx(x, y)] = 1.0f / phi(t);
      w_2[idx(x, y)] = 1.0f / phi(motion_obstruction_deriv_sq[idx(x, y)]);
      w_3[idx(x, y)] = 1.0f / phi(motion_background_deriv_sq[idx(x, y)]);
    }
  }
}

// Optimizing img_obstruction, img_background, and alpha, holding the rest constant
void optimizeImageComponents(int width, int height, float * img, float * img_obstruction, float * img_background, float * alpha, glm::vec2 * motion_obstruction, glm::vec2 * motion_background) {
  const int ROUNDS = 10;
  const float LEARNING_RATE = 0.1f;
  const float LAMBDA_1 = 1.0f;
  const float LAMBDA_2 = 1.0f;
  const float LAMBDA_3 = 1.0f;
  const float LAMBDA_P = 10e5;
  for (int i = 0; i < ROUNDS; i++) {
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        img_obstruction_grad[idx(x, y)] = 0.0f;
        alpha_grad[idx(x, y)] = 0.0f;
        img_background_grad[idx(x, y)] = 0.0f;
      }
    }
    // TODO
    computeDerivatives(img, img_obstruction, img_background, alpha, motion_obstruction, motion_background);
    computeWs1(img, img_obstruction, img_background, alpha, motion_obstruction, motion_background);
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        float dataTerm = 
          img[idx(x, y)] 
          - img_obstruction[warp(motion_obstruction, x, y)] 
          - alpha[warp(motion_obstruction, x, y)] * img_background[warp(motion_background, x, y)];

        img_obstruction_grad[warp(motion_obstruction, x, y)] -= dataTerm * w_1[idx(x, y)];
        alpha_grad[warp(motion_obstruction, x, y)] -= dataTerm * img_background[warp(motion_background, x, y)] * w_1[idx(x, y)];
        img_background_grad[warp(motion_background, x, y)] -= dataTerm * alpha[warp(motion_obstruction, x, y)] * w_1[idx(x, y)];

        // TODO: Check this?
        alpha_grad[idx(x, y)] += alpha[idx(x + 2, y)] - alpha[idx(x - 2, y)] + alpha[idx(x, y + 2)] - alpha[idx(x, y - 2)];

        img_obstruction_grad[idx(x, y)] += (LAMBDA_2 * w_2[idx(x + 1, y)] + LAMBDA_3 * img_obstruction_deriv_sq[idx(x + 1, y)]) 
          * (img_obstruction[idx(x + 2, y)] - img_obstruction[idx(x, y)]);
        img_obstruction_grad[idx(x, y)] += (LAMBDA_2 * w_2[idx(x - 1, y)] + LAMBDA_3 * img_obstruction_deriv_sq[idx(x - 1, y)])
          * (img_obstruction[idx(x, y)] - img_obstruction[idx(x - 2, y)]);
        img_obstruction_grad[idx(x, y)] += (LAMBDA_2 * w_2[idx(x, y + 1)] + LAMBDA_3 * img_obstruction_deriv_sq[idx(x, y + 1)])
          * (img_obstruction[idx(x, y + 2)] - img_obstruction[idx(x, y)]);
        img_obstruction_grad[idx(x, y)] += (LAMBDA_2 * w_2[idx(x, y - 1)] + LAMBDA_3 * img_obstruction_deriv_sq[idx(x, y - 1)])
          * (img_obstruction[idx(x, y)] - img_obstruction[idx(x, y - 2)]);

        img_background_grad[idx(x, y)] += LAMBDA_2 * w_3[idx(x + 1, y)] * (img_background[idx(x + 2, y)] - img_background[idx(x, y)]);
        img_background_grad[idx(x, y)] += LAMBDA_2 * w_3[idx(x - 1, y)] * (img_background[idx(x, y)] - img_background[idx(x - 2, y)]);
        img_background_grad[idx(x, y)] += LAMBDA_2 * w_3[idx(x, y + 1)] * (img_background[idx(x, y + 2)] - img_background[idx(x, y)]);
        img_background_grad[idx(x, y)] += LAMBDA_2 * w_3[idx(x, y - 1)] * (img_background[idx(x, y)] - img_background[idx(x, y - 2)]);

        if (img_obstruction[idx(x, y)] < 0.0f) {
          img_obstruction_grad[idx(x, y)] += LAMBDA_P * img_obstruction[idx(x, y)];
        }
        if (img_obstruction[idx(x, y)] > 1.0f) {
          img_obstruction_grad[idx(x, y)] += LAMBDA_P * (1.0f - img_obstruction[idx(x, y)]);
        }
        if (img_background[idx(x, y)] < 0.0f) {
          img_background_grad[idx(x, y)] += LAMBDA_P * img_background[idx(x, y)];
        }
        if (img_background[idx(x, y)] > 1.0f) {
          img_background_grad[idx(x, y)] += LAMBDA_P * (1.0f - img_background[idx(x, y)]);
        }
        if (alpha[idx(x, y)] < 0.0f) {
          alpha_grad[idx(x, y)] += LAMBDA_P * alpha[idx(x, y)];
        }
        if (alpha[idx(x, y)] > 1.0f) {
          alpha_grad[idx(x, y)] += LAMBDA_P * (1.0f - alpha[idx(x, y)]);
        }
      }
    }
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        img_obstruction[idx(x, y)] -= LEARNING_RATE * img_obstruction_grad[idx(x, y)];
        img_background[idx(x, y)] -= LEARNING_RATE * img_background_grad[idx(x, y)];
        alpha[idx(x, y)] -= LEARNING_RATE * alpha_grad[idx(x, y)];
      }
    }
  }
}

void optimizeMotionFields(int width, int height, float * img, float * img_obstruction, float * img_background, float * alpha, glm::vec2 * motion_obstruction, glm::vec2 * motion_background) {
  const int ROUNDS = 10;
  const float LEARNING_RATE = 0.1f;
  const float LAMBDA_4 = 1.0f;
  const float LAMBDA_5 = 1.0f;
  const float LAMBDA_P = 10e5;
  for (int i = 0; i < ROUNDS; i++) {
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        motion_obstruction_grad[idx(x, y)] = glm::vec2(0.0f, 0.0f);
        motion_background_grad[idx(x, y)] = glm::vec2(0.0f, 0.0f);;
      }
    }
    computeDerivatives(img, img_obstruction, img_background, alpha, motion_obstruction, motion_background);
    computeWs2(img, img_obstruction, img_background, alpha, motion_obstruction, motion_background);
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        float dataTerm = w_1[idx(x, y)] * 
          (img[idx(x, y)] 
           - img_obstruction[warp(motion_obstruction, x, y)]
           - alpha[warp(motion_obstruction, x, y)] * img_background[warp(motion_background, x, y)]);

        motion_obstruction_grad[idx(x, y)] -= dataTerm * img_obstruction_deriv[warp(motion_obstruction, x, y)];

        motion_background_grad[idx(x, y)] -= dataTerm * alpha[warp(motion_obstruction, x, y)] * img_background_deriv[warp(motion_obstruction, x, y)];

        motion_obstruction_grad[idx(x, y)] -= dataTerm * img_background[idx(x, y)] * alpha_deriv[warp(motion_obstruction, x, y)];

        motion_obstruction_grad[idx(x, y)] += LAMBDA_4 * w_2[idx(x + 1, y)] * (motion_obstruction[idx(x + 2, y)] - motion_obstruction[idx(x, y)]);
        motion_obstruction_grad[idx(x, y)] += LAMBDA_4 * w_2[idx(x - 1, y)] * (motion_obstruction[idx(x, y)] - motion_obstruction[idx(x - 2, y)]);
        motion_obstruction_grad[idx(x, y)] += LAMBDA_4 * w_2[idx(x, y + 1)] * (motion_obstruction[idx(x, y + 2)] - motion_obstruction[idx(x, y)]);
        motion_obstruction_grad[idx(x, y)] += LAMBDA_4 * w_2[idx(x, y - 1)] * (motion_obstruction[idx(x, y)] - motion_obstruction[idx(x, y - 2)]);

        motion_background_grad[idx(x, y)] += LAMBDA_4 * w_3[idx(x + 1, y)] * (motion_background[idx(x + 2, y)] - motion_background[idx(x, y)]);
        motion_background_grad[idx(x, y)] += LAMBDA_4 * w_3[idx(x - 1, y)] * (motion_background[idx(x, y)] - motion_background[idx(x - 2, y)]);
        motion_background_grad[idx(x, y)] += LAMBDA_4 * w_3[idx(x, y + 1)] * (motion_background[idx(x, y + 2)] - motion_background[idx(x, y)]);
        motion_background_grad[idx(x, y)] += LAMBDA_4 * w_3[idx(x, y - 1)] * (motion_background[idx(x, y)] - motion_background[idx(x, y - 2)]);
      }
    }
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        motion_obstruction[idx(x, y)] -= LEARNING_RATE * motion_obstruction_grad[idx(x, y)];
        motion_background[idx(x, y)] -= LEARNING_RATE * motion_background_grad[idx(x, y)];
      }
    }
  }
}

void optimizeImageParameters(int width, int height, float * img, float * img_obstruction, float * img_background, float * alpha, glm::vec2 * motion_obstruction, glm::vec2 * motion_background) {
  const int ROUNDS = 10;
  initializeBuffers(width, height);
  for (int i = 0; i < ROUNDS; i++) {
    optimizeImageComponents(width, height, img, img_obstruction, img_background, alpha, motion_obstruction, motion_background);
    optimizeMotionFields(width, height, img, img_obstruction, img_background, alpha, motion_obstruction, motion_background);
  }
  return;
}
