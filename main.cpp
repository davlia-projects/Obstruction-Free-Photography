#include <bits/stdc++.h>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
}
using namespace std;

int decode(AVCodecContext *avctx, AVFrame *frame, int *got_frame, AVPacket *pkt) {
  /*int ret;
  *got_frame = 0;
  if (pkt) {
    ret = avcodec_send_packet(avctx, pkt);
    if (ret < 0) {
      return ret == AVERROR_EOF ? 0 : ret;
    }
  }

  ret = avcodec_receive_frame(avctx, frame);
  if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
    return ret;
  }
  if (ret >= 0) {
    *got_frame = 1;
  }
  return 0;*/
	return avcodec_decode_video2(avctx, frame, got_frame, pkt);
}

int main(int argc, char * argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s [filename]\n", argv[0]);
    return 2;
  }

  av_register_all();
  avdevice_register_all();
  avcodec_register_all();

  AVCodec * codec;
  AVCodecContext * ctx = NULL;
  AVFormatContext * fctx = avformat_alloc_context();
  AVFrame * frame;
  AVPacket avpkt;

  if (avformat_open_input(&fctx, argv[1], NULL, NULL) != 0) {
    fprintf(stderr, "avformat_open_input failed\n");
    return 1;
  }
  int videoStream = 0;
  av_dump_format(fctx, 0, argv[1], 0);
  // AVCodecParameters * codecpar = fctx->streams[videoStream]->codecpar; 
  // codec = avcodec_find_decoder(codecpar->codec_id);
	ctx = fctx->streams[videoStream]->codec;
	codec = avcodec_find_decoder(ctx->codec_id);
  if (codec == NULL) {
    fprintf(stderr, "cannot find codec\n");
    return 1;
  }

  // ctx = avcodec_alloc_context3(codec);
  // avcodec_parameters_to_context(ctx, codecpar);
  if (avcodec_open2(ctx, codec, NULL) < 0) {
    fprintf(stderr, "cannot open codec\n");
    return 1;
  }
  frame = av_frame_alloc();

  AVFrame * rgb_frame = av_frame_alloc();
  AVPixelFormat pixfmt = AV_PIX_FMT_BGR24;
  int numBytes = av_image_get_buffer_size(pixfmt, ctx->width, ctx->height, 32);
  printf("bytes: %d\n", numBytes);
  uint8_t *buffer = (uint8_t *) av_malloc(numBytes * sizeof(uint8_t));
  int ret = 0;
  if ((ret = av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, buffer, pixfmt, ctx->width, ctx->height, 32)) < 0) {
    fprintf(stderr, "Cannot fill image (%d)\n", ret);
    return 1;
  }

  int ctr = 0;
  auto startTime = chrono::high_resolution_clock::now();
  while (av_read_frame(fctx, &avpkt) >= 0) {
    if (avpkt.stream_index == videoStream) {
      int frameFinished = 0;
      decode(ctx, frame, &frameFinished, &avpkt);
      if (frameFinished) {
        // Do something
        SwsContext * img_convert_ctx = sws_getCachedContext(
            NULL, ctx->width, ctx->height, ctx->pix_fmt, 
            ctx->width, ctx->height, pixfmt, SWS_BICUBIC, NULL, NULL, NULL);
        sws_scale(
            img_convert_ctx, frame->data, frame->linesize, 0, ctx->height,
            rgb_frame->data, rgb_frame->linesize);

        // TODO: rgb_frame->data[0] now correctly contains a BGR image
        ctr++;
        if (ctr % 100 == 0) {
          auto curTime = chrono::high_resolution_clock::now();
          chrono::duration<float> diff = curTime - startTime;
          float secs = diff.count();
          printf("FPS: %f\n", (float)ctr / secs);
        }
        av_packet_unref(&avpkt);
      }
    }
  }

  av_packet_unref(&avpkt);
  avcodec_close(ctx);
  av_free(frame);
  av_free(rgb_frame);
  avformat_close_input(&fctx);
  return 0;
}
