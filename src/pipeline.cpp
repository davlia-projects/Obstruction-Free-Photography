#include "pipeline.h"
#include <bits/stdc++.h>

using namespace std;

VideoProcessor::VideoProcessor(Pipeline * pipeline, char * inputFile, char * outputFile) {
  av_register_all();
  avdevice_register_all();
  avcodec_register_all();

  this->initInput(inputFile);
  this->initOutput(outputFile);
  this->initIntermediate(pipeline);

  this->frameCounter = 0;
}

VideoProcessor::~VideoProcessor() {
  this->cleanupInput();
  this->cleanupOutput();
}

int VideoProcessor::processFrame() {
  AVPacket inputPacket;
  int ret = av_read_frame(this->formatCtx, &inputPacket);
  if (ret < 0) {
    return ret;
  }
  if (inputPacket.stream_index != this->videoStream) {
    return ret;
  }

  int frameFinished = 0;
  avcodec_decode_video2(this->inputCtx, this->inputFrame, &frameFinished, &inputPacket);
  if (!frameFinished) {
    return ret;
  }
  SwsContext * inputConvertCtx = sws_getCachedContext(
      NULL, this->inputCtx->width, this->inputCtx->height, this->inputCtx->pix_fmt,
      this->inputCtx->width, this->inputCtx->height, this->intermediatePixelFormat,
      SWS_BICUBIC, NULL, NULL, NULL);
  sws_scale(inputConvertCtx, this->inputFrame->data, this->inputFrame->linesize, 0, 
      this->inputCtx->height, this->intermediateFrame->data, this->intermediateFrame->linesize);

  ret = this->pipeline->processFrame(this->intermediateFrame->data[0]);
  if (ret < 0) {
    return ret;
  } else if (ret == 0) {
    SwsContext * outputConvertCtx = sws_getCachedContext(
        NULL, this->outputCtx->width, this->outputCtx->height, this->intermediatePixelFormat,
        this->outputCtx->width, this->outputCtx->height, VideoProcessor::OUTPUT_PIXEL_FORMAT,
        SWS_BICUBIC, NULL, NULL, NULL);
    sws_scale(outputConvertCtx, this->intermediateFrame->data, this->intermediateFrame->linesize, 0,
        this->outputCtx->height, this->outputFrame->data, this->outputFrame->linesize);
    this->outputFrame->width = this->outputCtx->width;
    this->outputFrame->height = this->outputCtx->height;
    this->outputFrame->format = this->outputCtx->pix_fmt;
    this->outputFrame->pts = this->frameCounter;

    AVPacket outputPacket;
    av_init_packet(&outputPacket);
    outputPacket.data = NULL;
    outputPacket.size = 0;
    int gotOutput = 0;
    int outputSize = avcodec_encode_video2(this->outputCtx, &outputPacket, this->outputFrame, &gotOutput);
    if (outputSize < 0) {
      fprintf(stderr, "Error encoding frame %d\n", this->frameCounter);
      return -1;
    }
    if (gotOutput) {
      fwrite(outputPacket.data, 1, outputPacket.size, this->outputFilePtr);
      av_packet_unref(&outputPacket);
    }

    this->frameCounter++;
    av_packet_unref(&inputPacket);
  }
  return 0;
}

void VideoProcessor::initInput(char * inputFile) {
  this->formatCtx = avformat_alloc_context();
  if (avformat_open_input(&this->formatCtx, inputFile, NULL, NULL) != 0) {
    fprintf(stderr, "avformat_open_input failed\n");
    exit(1);
  }
  av_dump_format(this->formatCtx, 0, inputFile, 0);
  // TODO: may want to figure out how to derive this video stream instead of hardcoding
  this->videoStream = 0;

  this->inputCtx = formatCtx->streams[videoStream]->codec;
  this->inputCodec = avcodec_find_decoder(inputCtx->codec_id);
  if (this->inputCodec == NULL) {
    fprintf(stderr, "Could not find codec\n");
    exit(1);
  }
  if (avcodec_open2(this->inputCtx, this->inputCodec, NULL) < 0) {
    fprintf(stderr, "Could not open input codec\n");
    exit(1);
  }

  this->inputFrame = av_frame_alloc();
  return;
}

void VideoProcessor::cleanupInput() {
  avcodec_close(this->inputCtx);
  av_free(this->inputFrame);
  avformat_close_input(&this->formatCtx);
  return;
}

void VideoProcessor::initOutput(char * outputFile) {
  this->outputCodec = avcodec_find_encoder(VideoProcessor::OUTPUT_CODEC);
  this->outputCtx = avcodec_alloc_context3(this->outputCodec);
  this->outputCtx->bit_rate = this->inputCtx->bit_rate;
  this->outputCtx->width = this->inputCtx->width;
  this->outputCtx->height = this->inputCtx->height;
  // TODO: see if there's a way to derive this from the input
  this->outputCtx->time_base = (AVRational){1, 25};
  this->outputCtx->gop_size = this->inputCtx->gop_size;
  this->outputCtx->max_b_frames = this->inputCtx->max_b_frames;
  this->outputCtx->pix_fmt = VideoProcessor::OUTPUT_PIXEL_FORMAT;
  if (avcodec_open2(outputCtx, outputCodec, NULL) < 0) {
    fprintf(stderr, "Cannot open output codec\n");
    exit(1);
  }

  this->outputFilePtr = fopen(outputFile, "wb");
  if (!this->outputFilePtr) {
    fprintf(stderr, "Cannot open output file %s\n", outputFile);
    exit(1);
  }
  this->outputFrame = av_frame_alloc();
  int outBytes = av_image_get_buffer_size(this->outputCtx->pix_fmt, this->outputCtx->width, 
      this->outputCtx->height, VideoProcessor::OUTPUT_ALIGN);
  // TODO: do we need to free this somewhere?
  uint8_t * outBuffer = (uint8_t *) av_malloc(outBytes * sizeof(uint8_t));
  int ret = 0;
  if ((ret = av_image_fill_arrays(outputFrame->data, outputFrame->linesize, outBuffer,
          this->outputCtx->pix_fmt, this->outputCtx->width, this->outputCtx->height, 
          VideoProcessor::OUTPUT_ALIGN)) < 0) {
    fprintf(stderr, "Cannot fill image (error %d)\n", ret);
    exit(1);
  }
  return;
}

void VideoProcessor::cleanupOutput() {
  // MPEG1 encoding end code
  uint8_t endcode[] = {0, 0, 1, 0xb7};
  fwrite(endcode, 1, sizeof(endcode), this->outputFilePtr);
  fclose(this->outputFilePtr);
  
  avcodec_close(this->outputCtx);
  av_free(this->outputFrame);
}

void VideoProcessor::initIntermediate(Pipeline * pipeline) {
  this->pipeline = pipeline;
  this->intermediatePixelFormat = pipeline->getPixelFormat();
  this->intermediateFrame = av_frame_alloc();
  int intermediateBytes = av_image_get_buffer_size(this->intermediatePixelFormat, this->inputCtx->width, this->inputCtx->height, VideoProcessor::OUTPUT_ALIGN);
  uint8_t * intermediateBuffer = (uint8_t *) av_malloc(intermediateBytes * sizeof(uint8_t));
  int ret;
  if ((ret = av_image_fill_arrays(this->intermediateFrame->data, this->intermediateFrame->linesize, 
          intermediateBuffer, this->intermediatePixelFormat, this->inputCtx->width, this->inputCtx->height, VideoProcessor::OUTPUT_ALIGN)) < 0) {
    fprintf(stderr, "Cannot fill intermediate image (error %d)\n", ret);
    exit(1);
  }
}

void VideoProcessor::cleanupIntermediate() {
  av_free(this->intermediateFrame);
}
