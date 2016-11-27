//#include <cuda.h>
#include <bits/stdc++.h>
using namespace std;

void init(int width, int height) {}
void cleanup() {}

void blurFrame(uint8_t * dst, uint8_t * src, int width, int height) {
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
            r += src[(ty * width + tx) * 3] * kernel[k][l];
            g += src[(ty * width + tx) * 3 + 1] * kernel[k][l];
            b += src[(ty * width + tx) * 3 + 2] * kernel[k][l];
          }
        }
      }
      dst[(j * width + i) * 3] = r;
      dst[(j * width + i) * 3 + 1] = g;
      dst[(j * width + i) * 3 + 2] = b;
    }
  }
  return;
}
