#include <bits/stdc++.h>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
}

#include "pipeline.h"
#include "blank.h"
using namespace std;


int main(int argc, char * argv[]) {
  if (argc != 3) {
    printf("Usage: %s [input file] [output file]\n", argv[0]);
    return 1;
  }
  Pipeline * pipeline = new BlankPipeline();
  VideoProcessor * processor = new VideoProcessor(pipeline, argv[1], argv[2]);
  while (processor->processFrame() >= 0);
  return 0;
}
