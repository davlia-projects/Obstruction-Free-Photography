#include "sparse_separation.h"

void separatePoints(int width, int height, bool * pointGroup1, bool * pointGroup2, bool * sparseMap, glm::ivec2 * points) {
  glm::vec2 t1 = glm::vec2(0.0f, 0.0f);
  glm::vec2 t2 = glm::vec2(1.0f, 1.0f);
  int N = width * height;
  const int ITERATIONS = 50;
  for (int j = 0; j < ITERATIONS; j++) {
    memset(pointGroup1, 0, N * sizeof(bool));
    memset(pointGroup2, 0, N * sizeof(bool));
    glm::vec2 s1 = glm::vec2(0.0f, 0.0f);
    glm::vec2 s2 = glm::vec2(0.0f, 0.0f);
    float ct1 = 0.0f;
    float ct2 = 0.0f;
    for (int i = 0; i < N; i++) {
      if (!sparseMap[i]) {
        continue;
      }
      float d1 = (t1.x - points[i].x) * (t1.x - points[i].x) + (t1.y - points[i].y) * (t1.y - points[i].y);
      float d2 = (t2.x - points[i].x) * (t2.x - points[i].x) + (t2.y - points[i].y) * (t2.y - points[i].y);
      if (d1 < d2) {
        pointGroup1[i] = true;
        s1 += points[i];
        ct1 += 1.0f;
      } else {
        pointGroup2[i] = true;
        s2 += points[i];
        ct2 += 1.0f;
      }
      t1 = s1 / ct1;
      t2 = s2 / ct2;
    }
  }
  return;
}
