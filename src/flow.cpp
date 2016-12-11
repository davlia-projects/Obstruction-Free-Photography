#include <bits/stdc++.h>
#include "flow.h"

#define ITERS 10
#define RADIUS 4
#define LABELS 16
#define LAMBDA 20
#define SMOOTHNESS 2
enum DIR {LEFT, RIGHT, UP, DOWN, DATA};


struct MSG {
  unsigned int msg[5][LABELS];
  int best;
};

unsigned int smooth(int i, int j) {
  int delta = abs(i - j);
  return LAMBDA * (delta > SMOOTHNESS ? delta : SMOOTHNESS);
}

unsigned int datacost(int N, int width, int height, unsigned char * ref, unsigned char * tar, int x, int y, int l) {
  int sum = 0;
  for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
    for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
      int lx = l / (2 * RADIUS + 1) - RADIUS;
      int ly = l % (2 * RADIUS + 1) - RADIUS;
      int refPixel = ref[(y + dy) * width + x + dx];
      int tarPixel = ref[(y + dy + ly) * width + x + dx + lx];
      sum += abs(refPixel - tarPixel);
    }
  }
  unsigned int avg = sum / LABELS;
  return avg;
}

void init(int N, int width, int height, unsigned char * ref, unsigned char * tar, MSG * mrf) {
  printf("Init...\n");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < LABELS; ++k) {
        mrf[i].msg[j][k] = 0;
      }
    }
  }
  for (int y = RADIUS; y < height - RADIUS; ++y) {
    for (int x = RADIUS; x < width - RADIUS; ++x) {
      for (int l = 0; l < LABELS; ++l) {
        mrf[y * width + x].msg[DATA][l] = datacost(N, width, height, ref, tar, x, y, l);
      }
    }
  }
  printf("Init complete\n");
}

void send(int N, int width, int height, MSG * mrf, int x, int y, DIR dir) {
  unsigned int newMsg[LABELS];
  for (int i = 0; i < LABELS; ++i) {
    unsigned int min = UINT_MAX;
    for (int j = 0; j < LABELS; ++j) {
      unsigned int prob = 0;
      prob += smooth(i, j);
      prob += mrf[y * width + x].msg[DATA][j];
      if (dir != LEFT) {
          prob += mrf[y * width + x].msg[LEFT][j];
      }
      if (dir != RIGHT) {
          prob += mrf[y * width + x].msg[RIGHT][j];
      }
      if (dir != UP) {
          prob += mrf[y * width + x].msg[UP][j];
      }
      if (dir != DOWN) {
          prob += mrf[y * width + x].msg[DOWN][j];
      }
      (prob < min) && (min = prob);
    }
    newMsg[i] = min;
  }
  for (int i = 0; i < LABELS; ++i){
    switch (dir) {
      case LEFT:
          mrf[y * width + x - 1].msg[RIGHT][i] = newMsg[i];
          break;
      case RIGHT:
          mrf[y * width + x + 1].msg[LEFT][i] = newMsg[i];
          break;
      case UP:
          mrf[(y - 1) * width + x].msg[DOWN][i] = newMsg[i];
          break;
      case DOWN:
          mrf[(y + 1) * width + x].msg[UP][i] = newMsg[i];
          break;
      default:
          break;
    }
  }
}

void BP(int N, int width, int height, MSG * mrf, DIR dir) {
  printf("BP %d\n", dir);
  switch (dir) {
    case LEFT:
      for (int y = 0; y < height; ++y) {
        for (int x = width - 1; x >= 1 ; --x) {
          send(N, width, height, mrf, x, y, dir);
        }
      }
      break;
    case RIGHT:
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width - 1; ++x) {
          send(N, width, height, mrf, x, y, dir);
        }
      }
      break;
    case UP:
      for (int x = 0; x < width; ++x) {
        for (int y = height - 1; y >= 1; --y) {
          send(N, width, height, mrf, x, y, dir);
        }
      }
      break;
    case DOWN:
      for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height - 1; ++y) {
          send(N, width, height, mrf, x, y, dir);
        }
      }
      break;
  }
}

unsigned int MAP(int N, int width, int height, MSG * mrf) {
  for (int i = 0; i < N; ++i) {
    unsigned int best = UINT_MAX;
    for (int j = 0; j < LABELS; ++j) {
      unsigned int cost = 0;
      for (int k = 0; k < 5; ++k) {
        cost += mrf[i].msg[k][j];
      }
      if (cost < best) {
        best = cost;
        mrf[i].best = j;
      }
    }
  }
  unsigned int energy = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int label = mrf[y * width + x].best;
      energy += mrf[y * width + x].msg[DATA][label];
      if (x - 1 >= 0) {
        energy += smooth(label, mrf[y * width + x - 1].best);
      }
      if (x + 1 < width) {
        energy += smooth(label, mrf[y * width + x + 1].best);
      }
      if (y - 1 >= 0) {
        energy += smooth(label, mrf[(y - 1) * width + x].best);
      }
      if (y + 1 < height) {
        energy += smooth(label, mrf[(y + 1) * width + x].best);
      }
    }
  }
}


unsigned char * Flow::edgeFlow(int N, int width, int height, unsigned char * ref, unsigned char * tar) {
  MSG * mrf = new MSG[width * height];
  unsigned char * out = new unsigned char[width * height];
  init(N, width, height, ref, tar, mrf);
  for (int i = 0; i < ITERS; ++i) {
    BP(N, width, height, mrf, LEFT);
    BP(N, width, height, mrf, RIGHT);
    BP(N, width, height, mrf, UP);
    BP(N, width, height, mrf, DOWN);
    int energy = MAP(N, width, height, mrf);
    printf("Energy: %d\n", energy);
  }
  for (int y = RADIUS; y < height - RADIUS; ++y) {
    for (int x = RADIUS; x < width - RADIUS; ++x) {
      out[y * width + x] = mrf[y * width + x].best * (256.f / LABELS);
    }
  }
  return out;
}
