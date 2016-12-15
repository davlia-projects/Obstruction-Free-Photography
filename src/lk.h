#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

pair<vector<Point2f>, vector<Point2f>> lkEdgeFlow(int N, int width, int height, unsigned char * grad, unsigned char * img1, unsigned char * img2);
