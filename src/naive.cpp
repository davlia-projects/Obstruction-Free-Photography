#include "naive.h"

NaiveBlur::NaiveBlur(int width, int height) {
  this->width = width;
  this->height = height;
  this->tempBuffer = (uint8_t *) malloc(3 * width * height * sizeof(uint8_t));
}

int NaiveBlur::processFrame(uint8_t * frame) {
  int width = this->width;
  int height = this->height;
  float kernel[5][5] = {
    {0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
    {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
    {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
    {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
    {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
  };
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      float r = 0.0;
      float g = 0.0;
      float b = 0.0;
      for (int k = 0; k < 5; k++) {
        int tx = i + k - 2;
        for (int l = 0; l < 5; l++) {
          int ty = j + l - 2;
          if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
            r += frame[(ty * width + tx) * 3] * kernel[k][l];
            g += frame[(ty * width + tx) * 3 + 1] * kernel[k][l];
            b += frame[(ty * width + tx) * 3 + 2] * kernel[k][l];
          }
        }
      }
      tempBuffer[(j * width + i) * 3] = r;
      tempBuffer[(j * width + i) * 3 + 1] = g;
      tempBuffer[(j * width + i) * 3 + 2] = b;
    }
  }

  memcpy(frame, tempBuffer, 3 * this->width * this->height * sizeof(uint8_t));
  return 0;
}

AVPixelFormat NaiveBlur::getPixelFormat() {
  return AV_PIX_FMT_RGB24;
}
