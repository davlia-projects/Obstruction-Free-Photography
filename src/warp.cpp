#include <cstring>
#include <glm/vec2.hpp>
#include <algorithm>
#include <thrust/sort.h>
#include "interpolate.h"

#define NEIGHBORS 10
using namespace glm;
using namespace std;

typedef mot (pair<ivec2, ivec2>);

void transform(int N, int width, int height, mot * sparse, mot * dense) {

}

void warp(int N, int width, int height, mot * dense, unsigned char * image, unsigned char * warped) {
  for (int i = 0; i < N; i++) {
    int x = dense[idx].first.x;
    int y = dense[idx].first.y;
    int tx = x + dense[idx].second.x;
    int tx = y + dense[idx].second.y;
    if (0 <= tx && tx < width && 0 <= ty && ty < height) {
      warped[(tx * width) + ty] = image[x * width + y];
    }
  }
}
