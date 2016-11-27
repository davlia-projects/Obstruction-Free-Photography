#include <bits/stdc++.h>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
}

#include "basic_blur.h"
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
  if (argc != 3) {
    fprintf(stderr, "Usage: %s [input] [output]\n", argv[0]);
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

  if (avcodec_open2(ctx, codec, NULL) < 0) {
    fprintf(stderr, "cannot open codec\n");
    return 1;
  }
  frame = av_frame_alloc();
  AVPixelFormat pixfmt = AV_PIX_FMT_RGB24;

  AVCodec * out_codec = avcodec_find_encoder(AV_CODEC_ID_MPEG1VIDEO);
  AVCodecContext * out_ctx = avcodec_alloc_context3(out_codec);
  out_ctx->bit_rate = ctx->bit_rate;
  out_ctx->width = ctx->width;
  out_ctx->height = ctx->height;
  out_ctx->time_base = (AVRational){1, 25};
  out_ctx->gop_size = ctx->gop_size;
  out_ctx->max_b_frames = ctx->max_b_frames;
  out_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  if (avcodec_open2(out_ctx, out_codec, NULL) < 0) {
    fprintf(stderr, "cannot open encoder codec\n");
    return 1;
  }
  FILE * f = fopen(argv[2], "wb");
  if (!f) {
    fprintf(stderr, "cannot open output file %s\n", argv[2]);
    return 1;
  }
  AVPacket out_avpkt;
  AVFrame * out_frame = av_frame_alloc();
  int outBytes = av_image_get_buffer_size(out_ctx->pix_fmt, ctx->width, ctx->height, 32);
  uint8_t *outbuf = (uint8_t *) av_malloc(outBytes * sizeof(uint8_t));
  int ret = 0;
  if ((ret = av_image_fill_arrays(out_frame->data, out_frame->linesize, outbuf, out_ctx->pix_fmt, ctx->width, ctx->height, 32)) < 0) {
    fprintf(stderr, "Cannot fill out image (%d)\n", ret);
    return 1;
  }


  AVFrame * rgb_frame = av_frame_alloc();
  int numBytes = av_image_get_buffer_size(pixfmt, ctx->width, ctx->height, 32);
  printf("bytes: %d\n", numBytes);
  uint8_t *buffer = (uint8_t *) av_malloc(numBytes * sizeof(uint8_t));
  uint8_t *t_buffer = (uint8_t *) av_malloc(numBytes * sizeof(uint8_t));
  if ((ret = av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, buffer, pixfmt, ctx->width, ctx->height, 32)) < 0) {
    fprintf(stderr, "Cannot fill image (%d)\n", ret);
    return 1;
  }


	init(ctx->width, ctx->height);
  int ctr = 0;
  auto startTime = chrono::high_resolution_clock::now();
  while (av_read_frame(fctx, &avpkt) >= 0) {
    if (avpkt.stream_index == videoStream) {
      int frameFinished = 0;
      decode(ctx, frame, &frameFinished, &avpkt);
      if (frameFinished) {
        SwsContext * img_convert_ctx = sws_getCachedContext(
            NULL, ctx->width, ctx->height, ctx->pix_fmt, 
            ctx->width, ctx->height, pixfmt, SWS_BICUBIC, NULL, NULL, NULL);
        sws_scale(
            img_convert_ctx, frame->data, frame->linesize, 0, ctx->height,
            rgb_frame->data, rgb_frame->linesize);

        // TODO: rgb_frame->data[0] now correctly contains a RGB image
        // blurFrame(t_buffer, rgb_frame->data[0], ctx->width, ctx->height);
				if (ctr >= 2) {
					// memcpy(rgb_frame->data[0], t_buffer, numBytes * sizeof(uint8_t));
					
					// Encode
					SwsContext * out_convert_ctx = sws_getCachedContext(
							NULL, ctx->width, ctx->height, pixfmt,
							ctx->width, ctx->height, out_ctx->pix_fmt, SWS_BICUBIC, NULL, NULL, NULL);
					sws_scale(
							out_convert_ctx, rgb_frame->data, rgb_frame->linesize, 0, ctx->height,
							out_frame->data, out_frame->linesize);
					out_frame->width = ctx->width;
					out_frame->height = ctx->height;
					out_frame->format = out_ctx->pix_fmt;
					av_init_packet(&out_avpkt);
					out_avpkt.data = NULL;
					out_avpkt.size = 0;
					int got_output = 0;
					out_frame->pts = ctr;
					int out_size = avcodec_encode_video2(out_ctx, &out_avpkt, out_frame, &got_output);
					if (out_size < 0) {
						fprintf(stderr, "Error encoding frame %d\n", ctr);
						return 1;
					}
					if (got_output) {
						fwrite(out_avpkt.data, 1, out_avpkt.size, f);
						av_packet_unref(&out_avpkt);
					}
				}

        // Print stats
        ctr++;
        if (ctr % 10 == 0) {
          auto curTime = chrono::high_resolution_clock::now();
          chrono::duration<float> diff = curTime - startTime;
          float secs = diff.count();
          printf("FPS: %f\n", (float)ctr / secs);
        }
        av_packet_unref(&avpkt);
      }
    }
  }

	cleanup();

  // Sequence end code
  uint8_t endcode[] = {0, 0, 1, 0xb7};
  fwrite(endcode, 1, sizeof(endcode), f);
  fclose(f);

  av_packet_unref(&avpkt);
  avcodec_close(ctx);
  avcodec_close(out_ctx);
  av_free(frame);
  av_free(rgb_frame);
  avformat_close_input(&fctx);
  return 0;
}
