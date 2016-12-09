#include "canny.h"
#include <cstdio>
#include <cmath>

#define UPPERTHRESHOLD 60
#define LOWERTHRESHOLD 30

const float G_x[3 * 3] = {
  -1, 0, 1,
  -2, 0, 2,
  -1, 0, 1
};

const float G_y[3 * 3] = {
  1, 2, 1,
  0, 0, 0,
  -1, -2, -1
};

const float gaussian[5 * 5] = {
  2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159,
  4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
  5.f/159, 12.f/159, 15.f/159, 12.f/159, 2.f/159,
  4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
  2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159
};

void kernGrayscale(int N, unsigned char * in, unsigned char * out) {
  float avg = 0;
  for (int i = 0; i < N; i += 3) {
    avg = 0;
    avg += in[i];
    avg += in[i+1];
    avg += in[i+2];
    out[i/3] = avg / 3;
  }
}

// void kernConvolution(int N, unsigned char * in, unsigned char * out, int width, int height, const float * kernel, int kernSize) {
//   float r, g, b;
//   for (int x = 0; x < width; ++x) {
//     for (int y = 0; y < height; ++y) {
//     	r = g = b = 0.0;
//     	for (int i = 0; i < kernSize; i++) {
//     		int tx = x + i - kernSize/2;
//     		for (int j = 0; j < kernSize; j++) {
//     			int ty = y + j - kernSize/2;
//     			if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
//     				r += in[(ty * width + tx) * 3] * kernel[j * kernSize + i];
//     				g += in[(ty * width + tx) * 3 + 1] * kernel[j * kernSize + i];
//     				b += in[(ty * width + tx) * 3 + 2] * kernel[j * kernSize + i];
//     			}
//     		}
//     	}
//     	int idx = 3 * (y * width + x);
//     	out[idx] = r;
//     	out[idx + 1] = g;
//     	out[idx + 2] = b;
//     }
//   }
// }

void kernConvolution(int N, unsigned char * in, unsigned char * out, int width, int height, const float * kernel, int kernSize) {
  float c;
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      c = 0.0f;
    	for (int i = 0; i < kernSize; i++) {
    		int tx = x + i - kernSize/2;
    		for (int j = 0; j < kernSize; j++) {
    			int ty = y + j - kernSize/2;
    			if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
    				c += in[ty * width + tx] * kernel[j * kernSize + i];
    			}
    		}
    	}
    	out[y * width + x] = fabs(c);
      // if (c > 254 || c < 0)
      //   printf("%f %d\n", c, out[y * width + x]);
    }
  }
}

void kernGradient(int N, unsigned char * Gx, unsigned char * Gy, unsigned char * gradient, unsigned char * edgeDir) {
  float angle = 0, roundedAngle = 0;
  for (int i = 0; i < N; ++i) {
    gradient[i] = sqrt(Gx[i] * Gx[i] + Gy[i] * Gy[i]); // this might need to be a float
    angle = (atan2(Gx[i], Gy[i]) / 3.14159f) * 180.0f;
    if (((-22.5 < angle) && (angle <= 22.5)) || ((157.5 < angle) && (angle <= -157.5))) {
      roundedAngle = 0;
    }
		if (((-157.5 < angle) && (angle <= -112.5)) || ((22.5 < angle) && (angle <= 67.5))) {
      roundedAngle = 45;
    }
		if (((-112.5 < angle) && (angle <= -67.5)) || ((67.5 < angle) && (angle <= 112.5))) {
      roundedAngle = 90;
    }
		if (((-67.5 < angle) && (angle <= -22.5)) || ((112.5 < angle) && (angle <= 157.5))) {
      roundedAngle = 135;
    }
    edgeDir[i] = roundedAngle;
  }
}

void findEdge(unsigned char * gradient, unsigned char * edgeDir, unsigned char * out, int rowShift, int colShift, int row, int col, int dir, int width, int height, bool & edgeEnd) {
  int newRow;
  int newCol;
  unsigned long i;
  /* Find the row and column values for the next possible pixel on the edge */
	if (colShift < 0) {
		if (col > 0) {
      newCol = col + colShift;
    } else {
      edgeEnd = true;
    }
	} else if (col < width - 1) {
		newCol = col + colShift;
	} else {
    edgeEnd = true;		// If the next pixel would be off image, don't do the while loop
  }
	if (rowShift < 0) {
		if (row > 0) {
			newRow = row + rowShift;
		} else {
      edgeEnd = true;
    }
	} else if (row < height - 1) {
		newRow = row + rowShift;
	} else {
    edgeEnd = true;
  }

  while (edgeDir[newRow * width + newCol] == dir && !edgeEnd && gradient[newRow * width + newCol] > LOWERTHRESHOLD) {
    out[newRow * width + newCol] = 255;
    if (colShift < 0) {
			if (newCol > 0) {
        newCol = newCol + colShift;
      } else {
				edgeEnd = true;
      }
		} else if (newCol < width - 1) {
			newCol = newCol + colShift;
		} else {
      edgeEnd = true;
    }
		if (rowShift < 0) {
			if (newRow > 0) {
        newRow = newRow + rowShift;
      } else {
        edgeEnd = true;
      }
		} else if (newRow < height - 1) {
			newRow = newRow + rowShift;
		} else {
      edgeEnd = true;
    }
  }
}

void edgeTrace(int N, unsigned char * gradient, unsigned char * edgeDir, unsigned char * out, int width, int height) {
  bool edgeEnd;
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      edgeEnd = false;
      if (gradient[i * width + j] > UPPERTHRESHOLD) {
        switch(edgeDir[i * width + j]) {
          case 0:
            findEdge(gradient, edgeDir, out, 0, 1, i, j, 0, width, height, edgeEnd);
            break;
          case 45:
            findEdge(gradient, edgeDir, out, 1, 1, i, j, 45, width, height, edgeEnd);
            break;
          case 90:
            findEdge(gradient, edgeDir, out, 1, 0, i, j, 90, width, height, edgeEnd);
            break;
          case 135:
            findEdge(gradient, edgeDir, out, 1, -1, i, j, 135, width, height, edgeEnd);
            break;
          default:
            out[i * width + j] = 0;
            break;
        }
      } else {
        out[i * width + j] = 0;
      }
    }
  }
  /* Suppress any pixels not changed by the edge tracing */
	for (int i = 0; i < N; i++) {
    if (out[i] != 255 && out[i] != 0) {
      out[i] = 0;
    }
	}
}


// void nonMaxSuppression(int N, unsigned char * in, unsigned char * out, int width, int height) {
//   for (int i = 1; i < height - 1; ++i) {
//     for (int j = 1; j < width - 1; ++j) {
//       int angle = in[i * width + j];
//       switch(angle) {
//         case 0:
//           if (in[i * width + j] < in[i * width + j + 1] || in[i * width + j] < in[i * width + j - 1]) {
//             out[(i - 1) * width + j - 1] = 0;
//           }
//           break;
//         case 45:
//           if (in[i * width + j] < in[(i + 1) * width + j + 1] || in[i * width + j] < in[(i - 1) * width + j - 1]) {
//             out[(i - 1) * width + j - 1] = 0;
//           }
//           break;
//         case 90:
//           if (in[i * width + j] < in[(i + 1) * width + j] || in[i * width + j] < in[(i - 1) * width + j]) {
//             out[(i - 1) * width + j - 1] = 0;
//           }
//
//           break;
//         case 135:
//           if (in[i * width + j] < in[(i + 1) * width + j - 1] || in[i * width + j] < in[(i - 1) * width + j + 1]) {
//             out[(i - 1) * width + j - 1] = 0;
//           }
//           break;
//         default:
//           break;
//       }
//     }
//   }
// }

void nonMaxSuppression(int N, unsigned char * in, unsigned char * out, int width, int height) {
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      int angle = in[i * width + j];
      switch(angle) {
        case 0:
          if (in[i * width + j] < in[i * width + j + 1] || in[i * width + j] < in[i * width + j - 1]) {
            out[(i - 1) * width + j - 1] = 0;
          }
          break;
        case 45:
          if (in[i * width + j] < in[(i + 1) * width + j + 1] || in[i * width + j] < in[(i - 1) * width + j - 1]) {
            out[(i - 1) * width + j - 1] = 0;
          }
          break;
        case 90:
          if (in[i * width + j] < in[(i + 1) * width + j] || in[i * width + j] < in[(i - 1) * width + j]) {
            out[(i - 1) * width + j - 1] = 0;
          }

          break;
        case 135:
          if (in[i * width + j] < in[(i + 1) * width + j - 1] || in[i * width + j] < in[(i - 1) * width + j + 1]) {
            out[(i - 1) * width + j - 1] = 0;
          }
          break;
        default:
          break;
      }
    }
  }
}

void hysteresis(int N, unsigned char * in, unsigned char * out, int width, int height) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; i < width; j++) {
      int idx = i * width + j;
      out[idx] = in[idx];
      if (out[idx] > UPPERTHRESHOLD) {
        out[idx] = 255;
      } else if (out[idx] < LOWERTHRESHOLD) {
        out[idx] = 0;
      } else {
        bool anyHigh = false;
        bool anyBetween = false;
        for (int x = i - 1; x <= i + 1; ++x) {
          for (int y = j - 1; y <= j + 1; ++y) {
            if (x < 0 || y < 0 || x >= width || y >= height) {
              continue;
            }
            int xidx = x * width + y;
            if (out[xidx] > UPPERTHRESHOLD) {
              out[idx] = 255;
              anyHigh = true;
              break;
            } else if (out[xidx] <= UPPERTHRESHOLD && out[xidx] >= LOWERTHRESHOLD) {
              anyBetween = true;
            }
          }
          if (anyHigh) {
            break;
          }
        }
        if (!anyHigh && anyBetween) {
          for (int x = i - 2; x <= i + 2; x++) {
            for (int y = j - 2; j <= j + 2; j++) {
              if (x < 0 || y < 0 || x >= width || y >= height) {
                continue;
              }
              int xidx = x * width + y;
              if (out[xidx] > UPPERTHRESHOLD) {
                out[idx] = 255;
                anyHigh = true;
                break;
              }
            }
            if (anyHigh) {
              break;
            }
          }
          if(!anyHigh) {
            out[idx] = 0;
          }
        }
      }
    }
  }
}

void stride(int N, unsigned char * in, unsigned char * out) {
  for (int i = 0; i < N; i += 3) {
    out[i] = in[i / 3];
    out[i + 1] = in[i / 3];
    out[i + 2] = in[i / 3];
  }
}

unsigned char * Canny::edge(int N, unsigned char * in, int width, int height) {
  unsigned char * grayscale = new unsigned char[N / 3];
  unsigned char * smooth = new unsigned char[N / 3];
  unsigned char * Gx_mask = new unsigned char[N / 3];
  unsigned char * Gy_mask = new unsigned char[N / 3];
  unsigned char * gradient = new unsigned char[N / 3];
  unsigned char * edgeDir = new unsigned char[N / 3];

  kernGrayscale(N, in, grayscale);
  kernConvolution(N / 3, grayscale, smooth, width, height, gaussian, 5);
  kernConvolution(N / 3, smooth, Gx_mask, width, height, G_x, 3);
  kernConvolution(N / 3, smooth, Gy_mask, width, height, G_y, 3);
  kernGradient(N / 3, Gx_mask, Gy_mask, gradient, edgeDir);
  edgeTrace(N / 3, gradient, edgeDir, grayscale, width, height);
  // nonMaxSuppression(N / 3, edgeDir, grayscale, width, height);
  // hysteresis(N / 3, grayscale, smooth, width, height);

  unsigned char * out = new unsigned char[N];
  stride(N, grayscale, out);

  return out;
}
