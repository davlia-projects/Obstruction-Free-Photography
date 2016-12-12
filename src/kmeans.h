#pragma once

#include <bits/stdc++.h>
#include <glm/vec2.hpp>
using namespace std;

void separatePoints(int width, int height, bool * pointGroup1, bool * pointGroup2, bool * sparseMap, vector<glm::ivec2> * pointDiffs, int FRAMES, float THRESHOLD, float ITERATIONS);
//vector<pair<glm::ivec2, glm::ivec2>> copyPoints(int width, int height, bool * mask, glm::ivec2 * points);
