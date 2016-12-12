#include <iostream>
#include "CImg.h"
#include "canny.h"
#include "flow.h"
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
  CImg<unsigned char> images[5];
  images[0] = CImg<unsigned char>("img/hanoi_input_1.png");
  images[1] = CImg<unsigned char>("img/hanoi_input_2.png");
  images[2] = CImg<unsigned char>("img/hanoi_input_3.png");
  images[3] = CImg<unsigned char>("img/hanoi_input_4.png");
  images[4] = CImg<unsigned char>("img/hanoi_input_5.png");

  int width = images[0].width();
  int height = images[0].height();
  int N = width * height * 3;
  unsigned char * grayscale[5];
  unsigned char * in[5];
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
  edgeFlowPairs[0] = lkEdgeFlow(N / 3, width, height, gradient[0], grayscale[2], grayscale[3]);
  edgeFlowPairs[1] = lkEdgeFlow(N / 3, width, height, gradient[0], grayscale[2], grayscale[1]);

  /*printf("Generating second edge flow...\n");
  gradient[1] = new unsigned char[N / 3];
  memset(gradient[1], 0, (N / 3) * sizeof(unsigned char));
  gradient[2] = new unsigned char[N / 3];
  memset(gradient[2], 0, (N / 3) * sizeof(unsigned char));
  for (int i = 0; i < edgeFlowPairs[0].first.size(); i++) {
    Point2f nidx = edgeFlowPairs[0].first[i] + edgeFlowPairs[0].second[i];
    int nx = max(0, min(width - 1, (int)nidx.x));
    int ny = max(0, min(height - 1, (int)nidx.y));
    gradient[1][ny * width + nx] = 255;
  }
  for (int i = 0; i < edgeFlowPairs[1].first.size(); i++) {
    Point2f nidx = edgeFlowPairs[1].first[i] + edgeFlowPairs[1].second[i];
    int nx = max(0, min(width - 1, (int)nidx.x));
    int ny = max(0, min(height - 1, (int)nidx.y));
    gradient[2][ny * width + nx] = 255;
  }
  edgeFlowPairs[2] = lkEdgeFlow(N / 3, width, height, gradient[1], grayscale[3], grayscale[4]);
  edgeFlowPairs[3] = lkEdgeFlow(N / 3, width, height, gradient[2], grayscale[1], grayscale[0]);*/

  bool * group1 = new bool[N / 3];
  bool * group2 = new bool[N / 3];
  bool * sparseMap = new bool[N / 3];
  vector<glm::ivec2> * pointDiffs = new vector<glm::ivec2>[N / 3];
  memset(group1, 0, (N / 3) * sizeof(bool));
  memset(group2, 0, (N / 3) * sizeof(bool));
  memset(sparseMap, 0, (N / 3) * sizeof(bool));
  for (int i = 0; i < edgeFlowPairs[0].first.size(); i++) {
    Point2f & p = edgeFlowPairs[0].first[i];
    int idx = (int)p.y * width + (int)p.x;
    sparseMap[idx] = true;
    for (int j = 0; j < 2; j++) {
      Point2f & q = edgeFlowPairs[j].second[i];
      pointDiffs[idx].push_back(glm::ivec2(q.x, q.y));
    }
  }
  printf("Separating points...\n");
  separatePoints(width, height, group1, group2, sparseMap, pointDiffs, 2, 1000.0f, 30);

  unsigned char * flowViz = new unsigned char[N];
  memset(flowViz, 0, N * sizeof(unsigned char));
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      if (group1[idx]) {
        flowViz[3 * idx] = 255;
      } else if (group2[idx]) {
        flowViz[3 * idx + 1] = 255;
      }
    }
  }
  Mat mat(Size(width, height), CV_8UC3, flowViz);
  imwrite("flow.jpg", mat);

  /*for (int y = 0; y < image1.height(); y++) {
    for (int x = 0; x < image1.width(); x++) {
      if (group1[y * image1.width() + x]) {
        flowViz[3 * (y * image1.width() + x)] = 255;
      } else if (group2[y * image1.width() + x]) {
        flowViz[3 * (y * image1.width() + x) + 1] = 255;
      }
    }
  }
  Mat mat(Size(image1.width(), image1.height()), CV_8UC3, flowViz);
  imwrite("flow.jpg", mat);*/

  /*stride(N, flowViz, out1);
>>>>>>> 54ef39688d222cdf76d2be20a2332f5d05a5752d
  toIMG(out1, image1);
  CImgDisplay main_disp1(image1,"Click a point");
  while (!main_disp1.is_closed()) {
    main_disp1.wait();
  }*/
  // stride(N, gradient1, out1);
  // stride(N, gradient2, out2);
  // toIMG(out1, image1);
  // toIMG(out2, image2);
  //
  // CImgDisplay main_disp1(image1,"Click a point");
  // CImgDisplay main_disp2(image2,"Click a point");
  // while (!main_disp1.is_closed() && !main_disp2.is_closed()) {
  //   main_disp1.wait();
  //   main_disp2.wait();
  // }
  return 0;
}
