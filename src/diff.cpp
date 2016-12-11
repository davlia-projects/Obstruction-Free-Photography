#include "diff.h"

Diff::Diff(int width, int height) {
  this->width = width;
  this->height = height;
  int sz = 3 * width * height * sizeof(uint8_t);
  this->tempBuffer = (uint8_t *) malloc(sz);
  this->prevImage = (uint8_t *) malloc(sz);
  this->hasPrev = false;
}

Diff::~Diff() {
  free(this->tempBuffer);
  free(this->prevImage);
}

int Diff::processFrame(uint8_t * frame) {
  int sz = 3 * width * height * sizeof(uint8_t);
  if (!this->hasPrev) {
    this->hasPrev = true;
    memcpy(this->prevImage, frame, sz);
    return 0;
  }
  for (int i = 0; i < sz; i++) {
    uint8_t p = frame[i];
    frame[i] = p - this->prevImage[i];
    this->prevImage[i] = p;
  }
  return 0;
}

AVPixelFormat Diff::getPixelFormat() {
  return AV_PIX_FMT_RGB24;
}
