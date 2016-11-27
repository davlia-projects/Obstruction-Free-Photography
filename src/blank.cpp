#include "blank.h"

BlankPipeline::BlankPipeline() {
}

int BlankPipeline::processFrame(uint8_t * frame) {
  return 0;
}

AVPixelFormat BlankPipeline::getPixelFormat() {
  return AV_PIX_FMT_RGB24;
}
