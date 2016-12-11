#pragma once
#include <stdlib.h>
#include <stdint.h>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
}
using namespace std;

class Pipeline {
  public: 
    // Return a negative integer for error
    // Return a positive integer to indicate that there is no output, but we should keep feeding in frames
    virtual int processFrame(uint8_t * frame) = 0;

    virtual AVPixelFormat getPixelFormat() = 0;
};

class VideoProcessor {
  static const AVCodecID OUTPUT_CODEC = AV_CODEC_ID_MPEG1VIDEO;
  static const AVPixelFormat OUTPUT_PIXEL_FORMAT = AV_PIX_FMT_YUV420P;
  static const int OUTPUT_ALIGN = 32;

  private:
    void initInput(char * inputFile);
    void cleanupInput();
    AVFormatContext * formatCtx;
    AVCodecContext * inputCtx;
    AVCodec * inputCodec;
    AVFrame * inputFrame;
    int videoStream;

    void initOutput(char * outputFile);
    void cleanupOutput();
    AVCodecContext * outputCtx;
    AVCodec * outputCodec;
    AVFrame * outputFrame;
    FILE * outputFilePtr;

    void initIntermediate(Pipeline * pipeline);
    void cleanupIntermediate();
    Pipeline * pipeline;
    AVPixelFormat intermediatePixelFormat;
    AVFrame * intermediateFrame;

    int frameCounter;

  public:
    VideoProcessor(Pipeline * pipeline, char * inputFile, char * outputFile);
    ~VideoProcessor();
    int processFrame();
		int getFrameCounter();
};
