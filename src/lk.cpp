#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>

#include <bits/stdc++.h>

using namespace std;
using namespace cv;

vector<Point2f> lkEdgeFlow(int N, int width, int height, unsigned char * grad, unsigned char * img1, unsigned char * img2) {
  //printf("ASDF: %d %d %d\n", N, width, height);
  cout << "b " << N << "," << width << "," << height << endl;
  Mat mat1(width, height, CV_8UC1, img1, 1);
  Mat mat2(width, height, CV_8UC1, img2, 1);
  vector<Point2f> points[2];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (grad[y * width + x] == 255) {
        points[0].push_back(Point2f(x, y));
      }
    }
  }
  vector<uchar> status;
  vector<float> err;
  calcOpticalFlowPyrLK(mat1, mat2, points[0], points[1], status, err);
  printf("%d %d\n", points[0].size(), points[1].size());
  for (int i = 0; i < points[1].size(); i++) {
    points[1][i] -= points[0][i];
    printf("%f %f\n", points[1][i].x, points[1][i].y);
  }
  return points[1];
}
