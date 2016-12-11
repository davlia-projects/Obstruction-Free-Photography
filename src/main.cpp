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
  CImg<unsigned char> image1("img/hanoi_input_1.png");
  CImg<unsigned char> image2("img/hanoi_input_2.png");

  int N = image1.width() * image1.height() * 3;
  unsigned char * out1 = new unsigned char[N];
  unsigned char * out2 = new unsigned char[N];
  unsigned char * g1 = new unsigned char[N / 3];
  unsigned char * g2 = new unsigned char[N / 3];

  unsigned char * in1 = toRGB(image1);
  unsigned char * in2 = toRGB(image2);
  kernGrayscale(N, in1, g1);
  kernGrayscale(N, in2, g2);

  unsigned char * gradient1 = Canny::edge(N / 3, image1.width(), image1.height(), g1);
  unsigned char * gradient2 = Canny::edge(N / 3, image2.width(), image2.height(), g2);

  //unsigned char * flow = Flow::edgeFlow(N / 3, image1.width(), image1.height(), gradient1, gradient2);
  auto t = lkEdgeFlow(N / 3, image1.width(), image1.height(), gradient1, g1, g2);
  unsigned char * flowViz = new unsigned char[N];
  glm::ivec2 * flowPoints = new glm::ivec2[N / 3];
  bool * sparseMap = new bool[N / 3];
  bool * group1 = new bool[N / 3];
  bool * group2 = new bool[N / 3];
  memset(sparseMap, 0, (N / 3) * sizeof(bool));
  memset(flowViz, 0, N * sizeof(unsigned char));
  for (int i = 0; i < t.first.size(); i++) {
    Point2f & p = t.first[i];
    Point2f & d = t.second[i];
    //flowViz[(int)p.y * image1.width() + (int)p.x] = sqrt(d.x * d.x + d.y * d.y);
    sparseMap[(int)p.y * image1.width() + (int)p.x] = 1;
    flowPoints[(int)p.y * image1.width() + (int)p.x] = glm::ivec2((int)t.second[i].x, (int)t.second[i].y);
  }
  separatePoints(image1.width(), image1.height(), group1, group2, sparseMap, flowPoints);
  auto v1 = copyPoint(image1.width(), image1.height(), group1, flowPoints);
  auto v2 = copyPoint(image1.width(), image1.height(), group2, flowPoints);

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
