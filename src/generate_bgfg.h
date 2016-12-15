#pragma once

#include <glm/glm.hpp>
#include <utility>

void generateBgFg(int width, int height, int frames, float * bgImg, float * fgImg, unsigned char ** grayscale, std::pair<glm::vec2, glm::vec2> * groupVectors, float * bgPixels);
