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

const float gaussian[5 * 5] = {
  2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159,
  4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
  5.f/159, 12.f/159, 15.f/159, 12.f/159, 2.f/159,
  4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
  2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159
};

int main() {
  const float SCALE_1_255 = (1.0f / 255.0f);
  CImg<unsigned char> images[5];
  /*images[0] = CImg<unsigned char>("img/hanoi1.png");
  images[1] = CImg<unsigned char>("img/hanoi2.png");
  images[2] = CImg<unsigned char>("img/hanoi3.png");
  images[3] = CImg<unsigned char>("img/hanoi4.png");
  images[4] = CImg<unsigned char>("img/hanoi5.png");*/
  images[0] = CImg<unsigned char>("img/window1.png");
  images[1] = CImg<unsigned char>("img/window2.png");
  images[2] = CImg<unsigned char>("img/window3.png");
  images[3] = CImg<unsigned char>("img/window4.png");
  images[4] = CImg<unsigned char>("img/window5.png");

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
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        grayscale[i][idx] = images[i](x, y, 0);
      }
    }
    // in[i] = toRGB(images[i]);
    // kernGrayscale(N, in[i], grayscale[i]);
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
  vector<float> * bgPixels = new vector<float>[N / 3];
  float * bgImg = new float[N / 3];
  float * fgImg = new float[N / 3];
  float * alpha = new float[N / 3];
  memset(bgImg, 0, (N / 3) * sizeof(float));
  memset(fgImg, 0, (N / 3) * sizeof(float));
  memset(alpha, 0, (N / 3) * sizeof(float));
  for (int i = 0; i < N/3; i++) {
    fgImg[i] = bgImg[i] = grayscale[2][i] * 0.20f;
    bgImg[i] = 255.0f;
  }
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
    separatePoints(width, height, group1[j], group2[j], sparseMap, pointDiffs, 61.0f, 30);

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
        // bgImg[idx] += grayscale[nj][warpBg.y * width + warpBg.x] * 0.20f;
        // bgImg[idx] = max(bgImg[idx], (float)grayscale[nj][warpBg.y * width + warpBg.x]);
        bgImg[idx] = min(bgImg[idx], (float)grayscale[nj][warpBg.y * width + warpBg.x]);
        bgPixels[idx].push_back(grayscale[nj][warpBg.y * width + warpBg.x]);
        glm::ivec2 warpFg = glm::ivec2(x, y) + denseFg[j][idx].second;
        warpFg.x = max(0, min(width - 1, warpFg.x));
        warpFg.y = max(0, min(height - 1, warpFg.y));
        fgImg[idx] += grayscale[nj][warpFg.y * width + warpFg.x] * 0.20f;
        // fgImg[idx] = min(fgImg[idx], (float)grayscale[nj][warpFg.y * width + warpFg.x]);
      }
    }
  }

  // Spatial coherence
  /*
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      sort(bgPixels[idx].begin(), bgPixels[idx].end());
      bgImg[idx] = bgPixels[idx][2];
    }
  }
  
  for (int i = 0; i < 10; i++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        float mn = 1e9;
        int mj = 0;
        for (int j = 0; j < 5; j++) {
          float d = 0.0f;
          for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
              int nx = max(0, min(width - 1, x + dx));
              int ny = max(0, min(height - 1, y + dy));
              float pd = bgImg[ny * width + nx] - bgPixels[idx][j];
              d += fabs(pd);
            }
          }
          float sd = bgImg[y * width - x] - (float)grayscale[2][idx];
          d += fabs(sd) * 10;
          if (d < mn) {
            mn = d;
            mj = j;
          }
        }
        bgImg[idx] = bgPixels[idx][mj];
      }
    }
  }
  */

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
      if (fabs(bgImg[idx] - grayscale[2][idx]) < 25.0f) {
        alpha[idx] = 1.0f;
        alphaViz[idx] = 255;
      } else {
        alpha[idx] = 0.0f;
        alphaViz[idx] = 0;
      }
      if (alpha[idx] > 0.0f) {
        fgViz[idx] = 255;
        bgViz[idx] = grayscale[2][idx];
      } else {
        fgViz[idx] = (unsigned char) fgImg[idx];
      }
      bgImg[idx] *= SCALE_1_255;
      fgImg[idx] *= SCALE_1_255;
    }
  }
  Mat gs(Size(width, height), CV_8UC1, grayscale[2]);
  imwrite("grayscale.jpg", gs);
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
      alphaViz[idx] = (unsigned char) (alpha[idx] * 255.0f);
      if (alpha[idx] > 0.0f) {
        fgViz[idx] = 255;
      } else {
        fgViz[idx] = (unsigned char) (fgImg[idx] * 255.0f);
      }
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

  /*for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      float c = 0.0f;
      for (int i = 0; i < FRAMES; i++) {
        int ni = (i <= 2) ? i - 1 : i;
        if (i == 0) {
          ni = 2;
        }
        glm::vec2 v = VO[i * width * height + idx];
        int nx = max(0, min(width - 1, (int)(x + v.x)));
        int ny = max(0, min(height - 1, (int)(y + v.y)));
        c += grayscale[ni][ny * width + nx];
      }
      bgViz[idx] = c / 5.0f;
    }
  }
  Mat ffmat(Size(width, height), CV_8UC1, bgViz);
  imwrite("bg_img3.jpg", ffmat);*/

  return 0;
}
