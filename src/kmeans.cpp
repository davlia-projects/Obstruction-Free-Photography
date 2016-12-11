#include "kmeans.h"

/*void separatePoints(int width, int height, bool * pointGroup1, bool * pointGroup2, bool * sparseMap, glm::ivec2 * points) {
  //glm::vec2 t1 = glm::vec2(0.0f, 0.0f);
  //glm::vec2 t2 = glm::vec2(0.0f, 0.0f);
  float t1 = 0.0f;
  float t2 = 0.0f;
  float ct1 = 0.0f;
  float ct2 = 0.0f;
  int N = width * height;
  for (int i = 0; i < N; i++) {
    if (!sparseMap[i]) continue;
    if (rand() % 2 == 0) {
      t1 += sqrt(points[i].x * points[i].x + points[i].y * points[i].y);
      ct1 += 1.0f;
    } else {
      t2 += sqrt(points[i].x * points[i].x + points[i].y * points[i].y);
      ct2 += 1.0f;
    }
  }
  t1 /= ct1;
  t2 /= ct2;
  const int ITERATIONS = 50;
  for (int j = 0; j < ITERATIONS; j++) {
    memset(pointGroup1, 0, N * sizeof(bool));
    memset(pointGroup2, 0, N * sizeof(bool));
    //glm::vec2 s1 = glm::vec2(0.0f, 0.0f);
    //glm::vec2 s2 = glm::vec2(0.0f, 0.0f);
    float s1 = 0.0f;
    float s2 = 0.0f;
    ct1 = 0.0f;
    ct2 = 0.0f;
    bool doshow = true;
    for (int i = 0; i < N; i++) {
      if (!sparseMap[i]) {
        continue;
      }
      //float d1 = (t1.x - points[i].x) * (t1.x - points[i].x) + (t1.y - points[i].y) * (t1.y - points[i].y);
      //float d2 = (t2.x - points[i].x) * (t2.x - points[i].x) + (t2.y - points[i].y) * (t2.y - points[i].y);
      float d = sqrt(points[i].x * points[i].x + points[i].y * points[i].y);
      if (doshow) {
        printf("%d %f %f\n", i, fabs(d - t1), fabs(d - t2));
        doshow = false;
      }
      if (fabs(d - t1) < fabs(d - t2)) {
        pointGroup1[i] = true;
        s1 += d;
        ct1 += 1.0f;
      } else {
        pointGroup2[i] = true;
        s2 += d;
        ct2 += 1.0f;
      }
      if (ct1 < 0.01f) {
        ct1 = 1.0f;
      }
      if (ct2 < 0.01f) {
        ct2 = 1.0f;
      }
      t1 = s1 / ct1;
      t2 = s2 / ct2;
    }
  }
  return;
}*/
void separatePoints(int width, int height, bool * pointGroup1, bool * pointGroup2, bool * sparseMap, glm::ivec2 * points) {
  glm::vec2 t1 = glm::vec2(0.0f, 0.0f);
  float ct1 = 0.0f;
  int N = width * height;
  for (int i = 0; i < N; i++) {
    if (!sparseMap[i]) continue;
    if (rand() % 2 == 0) {
      t1 += points[i];
      ct1 += 1.0f;
    }
  }
  t1 /= ct1;
  const float THRESHOLD_1 = 450.0f;
  const int ITERATIONS = 30;
  for (int j = 0; j < ITERATIONS; j++) {
    memset(pointGroup1, 0, N * sizeof(bool));
    glm::vec2 s1 = glm::vec2(0.0f, 0.0f);
    ct1 = 0.0f;
    for (int i = 0; i < N; i++) {
      if (!sparseMap[i]) continue;
      glm::vec2 d = t1 - glm::vec2((float)points[i].x, (float)points[i].y);
      if ((d.x * d.x + d.y * d.y) <= THRESHOLD_1) {
        pointGroup1[i] = true;
        s1 += points[i];
        ct1 += 1.0f;
      }
    }
    t1 = s1 / ct1;
  }
  for (int i = 0; i < N; i++) {
    if (sparseMap[i] && !pointGroup1[i]) pointGroup2[i] = true;
  }
}

vector<pair<glm::ivec2, glm::ivec2>> copyPoints(int width, int height, bool * mask, glm::ivec2 * points) {
  vector<pair<glm::Ivec2, glm::ivec2>> ret;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (mask[y * width + x]) {
        ret.push_back(make_pair(glm::ivec2(x, y), points[y * width + x]));
      }
    }
  }
  return ret;
}
