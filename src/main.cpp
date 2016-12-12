#include <iostream>
#include "CImg.h"
#include "canny.h"
#include "flow.h"
#include "gradient_descent.h"
#include "interpolate.h"
#include "lk.h"
#include "kmeans.h"

using namespace cimg_library;
using namespace std;

unsigned char * toRGB(CImg<unsigned char> & image) {
  unsigned char * in = new unsigned char[image.size()];
  for (int y = 0; y < image.height(); ++y) {
    for (int x = 0; x < image.width(); ++x) {
      int idx = (image.width() * y + x) * 3;
      in[idx] = image(x, y, 0);
      in[idx + 1] = image(x, y, 1);
      in[idx + 2] = image(x, y, 2);
    }
  }
  return in;
}

void toIMG(unsigned char * in, CImg<unsigned char> & image) {
  for (unsigned int i = 0; i < image.size() / 3; ++i) {
    int y = i / image.width();
    int x = i % image.width();
    image(x, y, 0) = in[3 * i];
    image(x, y, 1) = in[3 * i + 1];
    image(x, y, 2) = in[3 * i + 2];
  }
}

void kernGrayscale(int N, unsigned char * in, unsigned char * out) {
  float avg = 0;
  for (int i = 0; i < N; i += 3) {
    avg = 0;
    avg += in[i];
    avg += in[i+1];
    avg += in[i+2];
    out[i / 3] = avg / 3;
  }
}

void stride(int N, unsigned char * in, unsigned char * out) {
  for (int i = 0; i < N; i += 3) {
    out[i] = in[i / 3];
    out[i + 1] = in[i / 3];
    out[i + 2] = in[i / 3];
  }
}

int main() {
  const float SCALE_1_255 = (1.0f / 255.0f);
  CImg<unsigned char> images[5];
  images[0] = CImg<unsigned char>("img/hanoi1.png");
  images[1] = CImg<unsigned char>("img/hanoi2.png");
  images[2] = CImg<unsigned char>("img/hanoi3.png");
  images[3] = CImg<unsigned char>("img/hanoi4.png");
  images[4] = CImg<unsigned char>("img/hanoi5.png");

  int width = images[0].width();
  int height = images[0].height();
  int N = width * height * 3;
  unsigned char * grayscale[5];
  unsigned char * in[5];
  bool * group1[5];
  bool * group2[5];
  unsigned char * gradient[3];
  pair<vector<Point2f>, vector<Point2f>> edgeFlowPairs[5];
  for (int i = 0; i < 5; i++) {
    grayscale[i] = new unsigned char[N / 3];
    in[i] = toRGB(images[i]);
    kernGrayscale(N, in[i], grayscale[i]);
  }

  printf("Detecting edges...\n");
  gradient[0] = Canny::edge(N / 3, width, height, grayscale[2]);
  printf("Generating edge flow...\n");
  edgeFlowPairs[0] = lkEdgeFlow(N / 3, width, height, gradient[0], grayscale[2], grayscale[0]);
  edgeFlowPairs[1] = lkEdgeFlow(N / 3, width, height, gradient[0], grayscale[2], grayscale[1]);
  edgeFlowPairs[2] = lkEdgeFlow(N / 3, width, height, gradient[0], grayscale[2], grayscale[3]);
  edgeFlowPairs[3] = lkEdgeFlow(N / 3, width, height, gradient[0], grayscale[2], grayscale[4]);

  glm::ivec2 * pointDiffs = new glm::ivec2[N / 3];
  bool * sparseMap = new bool[N / 3];
  pair<glm::ivec2, glm::ivec2> * denseBg[4];
  pair<glm::ivec2, glm::ivec2> * denseFg[4];
  float * bgImg = new float[N / 3];
  float * fgImg = new float[N / 3];
  float * alpha = new float[N / 3];
  memset(bgImg, 0, (N / 3) * sizeof(float));
  memset(fgImg, 0, (N / 3) * sizeof(float));
  memset(alpha, 0, (N / 3) * sizeof(float));
  for (int j = 0; j < 4; j++) {
    memset(sparseMap, 0, (N / 3) * sizeof(bool));
    for (int i = 0; i < edgeFlowPairs[j].first.size(); i++) {
      Point2f & p = edgeFlowPairs[j].first[i];
      int idx = (int)p.y * width + (int)p.x;
      sparseMap[idx] = true;
      Point2f & q = edgeFlowPairs[j].second[i];
      pointDiffs[idx] = glm::ivec2(q.x, q.y);
    }
    printf("Separating...\n");
    group1[j] = new bool[N / 3];
    group2[j] = new bool[N / 3];
    separatePoints(width, height, group1[j], group2[j], sparseMap, pointDiffs, 45.0f, 30);

    vector<pair<glm::ivec2, glm::ivec2>> fgPoints;
    vector<pair<glm::ivec2, glm::ivec2>> bgPoints;
    fgPoints.clear();
    bgPoints.clear();
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        if (group1[j][idx]) {
          bgPoints.push_back(make_pair(glm::ivec2(x, y), pointDiffs[idx]));
        } else if (group2[j][idx]) {
          fgPoints.push_back(make_pair(glm::ivec2(x, y), pointDiffs[idx]));
        }
      }
    }
    printf("Interpolating...\n");
    denseBg[j] = interpolate(bgPoints.size(), width, height, bgPoints.data());
    denseFg[j] = interpolate(fgPoints.size(), width, height, fgPoints.data());
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        int nj = (j >= 2) ? j + 1 : j;
        glm::ivec2 warpBg = glm::ivec2(x, y) + denseBg[j][idx].second;
        warpBg.x = max(0, min(width - 1, warpBg.x));
        warpBg.y = max(0, min(height - 1, warpBg.y));
        // bgImg[idx] += grayscale[nj][warpBg.y * width + warpBg.x];
        bgImg[idx] = max(bgImg[idx], (float)grayscale[nj][warpBg.y * width + warpBg.x]);
        glm::ivec2 warpFg = glm::ivec2(x, y) + denseFg[j][idx].second;
        warpFg.x = max(0, min(width - 1, warpFg.x));
        warpFg.y = max(0, min(height - 1, warpFg.y));
        // fgImg[idx] += grayscale[nj][warpFg.y * width + warpFg.x];
        fgImg[idx] = min(fgImg[idx], (float)grayscale[nj][warpFg.y * width + warpFg.x]);
      }
    }
  }

  unsigned char * bgViz = new unsigned char[N / 3];
  unsigned char * bgDelta = new unsigned char[N / 3];
  unsigned char * fgViz = new unsigned char[N / 3];
  unsigned char * alphaViz = new unsigned char[N / 3];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      // bgImg[idx] *= 0.25f;
      // fgImg[idx] *= 0.25f;
      bgViz[idx] = (unsigned char) bgImg[idx];
      fgViz[idx] = (unsigned char) fgImg[idx];
      bgImg[idx] *= SCALE_1_255;
      fgImg[idx] *= SCALE_1_255;
      if (fabs(bgImg[idx] - grayscale[2][idx]) < 35.0f) {
        alpha[idx] = 1.0f;
        alphaViz[idx] = 255;
      } else {
        alpha[idx] = 0.0f;
        alphaViz[idx] = 0;
      }
    }
  }
  Mat mat(Size(width, height), CV_8UC1, bgViz);
  imwrite("bg_img.jpg", mat);
  Mat mat2(Size(width, height), CV_8UC1, fgViz);
  imwrite("fg_img.jpg", mat2);
  Mat mat3(Size(width, height), CV_8UC1, alphaViz);
  imwrite("alpha_img.jpg", mat3);


  unsigned char * flowViz = new unsigned char[N];
  for (int i = 0; i < 4; i++) {
    memset(flowViz, 0, N * sizeof(unsigned char));
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        if (group1[i][idx]) {
          flowViz[3 * idx] = 255;
        } else if (group2[i][idx]) {
          flowViz[3 * idx + 1] = 255;
        }
      }
    }
    Mat mat(Size(width, height), CV_8UC3, flowViz);
    char str[40];
    sprintf(str, "flow%d.jpg", i);
    imwrite(str, mat);
  }

  int FRAMES = 5;
  glm::vec2 * VB = new glm::vec2[(N / 3) * FRAMES];
  glm::vec2 * VO = new glm::vec2[(N / 3) * FRAMES];
  float * sequence = new float[(N / 3) * FRAMES];
  for (int i = 0; i < FRAMES; i++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = i * width * height + y * width + x;
        int idx2 = y * width + x;
        if (i == 0) {
          VB[idx] = glm::vec2(0.0f, 0.0f);
          VO[idx] = glm::vec2(0.0f, 0.0f);
          sequence[idx] = (float)grayscale[2][idx2] * SCALE_1_255;
        } else {
          VB[idx] = glm::vec2(denseBg[i][idx2].second.x, denseBg[i][idx2].second.y);
          VO[idx] = glm::vec2(denseFg[i][idx2].second.x, denseFg[i][idx2].second.y);
          int ni = (i <= 2) ? i - 1 : i;
          sequence[idx] = (float)grayscale[ni][idx2] * SCALE_1_255;
        }
      }
    }
  }
  GradientDescent gd(width, height, 5, VB, VO, alpha, fgImg, bgImg, sequence);
  gd.optimize();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      bgDelta[idx] = bgViz[idx] - (unsigned char) (bgImg[idx] * 255.0f);
      bgViz[idx] = (unsigned char) (bgImg[idx] * 255.0f);
      fgViz[idx] = (unsigned char) (fgImg[idx] * 255.0f);
      alphaViz[idx] = (unsigned char) (alpha[idx] * 255.0f);
    }
  }
  Mat fmat(Size(width, height), CV_8UC1, bgViz);
  imwrite("bg_img2.jpg", fmat);
  Mat fmat2(Size(width, height), CV_8UC1, fgViz);
  imwrite("fg_img2.jpg", fmat2);
  Mat fmat3(Size(width, height), CV_8UC1, alphaViz);
  imwrite("alpha_img2.jpg", fmat3);
  Mat fmat4(Size(width, height), CV_8UC1, bgDelta);
  imwrite("bg_delta.jpg", fmat4);


  return 0;
}
