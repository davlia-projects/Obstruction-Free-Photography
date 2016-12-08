#include <iostream>
#include "CImg.h"
#include "canny.h"

using namespace cimg_library;

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

int main() {
  CImg<unsigned char> image("img/hanoi_input_1.png");
  unsigned char * in = toRGB(image);
  unsigned char * out = Canny::edge(image.size(), in, image.width(), image.height());
  toIMG(out, image);

  CImgDisplay main_disp(image,"Click a point");
  while (!main_disp.is_closed()) {
    main_disp.wait();
  }
  return 0;
}
