#include <cstring>
#include <glm/vec2.hpp>
#include <algorithm>
#include "warp.h"

#define NEIGHBORS 10
using namespace glm;
using namespace std;

typedef pair<ivec2, ivec2> mot;

void warp(int N, int width, int height, mot * dense, unsigned char * image, unsigned char * warped) {
  for (int i = 0; i < N; i++) {
    int x = dense[i].first.x;
    int y = dense[i].first.y;
    int tx = x + dense[i].second.x;
    int ty = y + dense[i].second.y;
    if (0 <= tx && tx < width && 0 <= ty && ty < height) {
      warped[(tx * width) + ty] = image[x * width + y];
    }
  }
}
