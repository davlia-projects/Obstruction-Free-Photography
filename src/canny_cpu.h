#pragma once

namespace Canny {
  unsigned char * edge(int N, int width, int height, unsigned char * in);
  void kernSmooth(int N, int width, int height, unsigned char * in, unsigned char * out, const float * kernel, int kernSize);
}
