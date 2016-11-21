#pragma once

void init(int width, int height);
void cleanup();
void blurFrame(uint8_t * dst, uint8_t * src, int width, int height);
