# CIS 565 Final Project Proposal

## CUDA-Accelerated Video Processing Pipeline

## Overview
Many livestreaming applications (Youtube, Facebook live, Twitch) have a need to be able to process video streams on the fly through compression, encoding, up/down sampling, stitching, and any other post-processing effects. Given the data parallel properties of some of these challenges, GPGPUs are most suited for the task. We intend to create a CUDA-accelerated real-time video processing pipeline for online use. We reference existing image processing pipelines and use it as inspiration for a novel video pipeline that will take advantage of the temporal locality of the data.

The tech stack we will be using for this project will primarily be C/C++ and CUDA. 

## Goals/Features
 * Upsample/Downsampling
 * [Frame interpolation](https://www.wikiwand.com/en/Motion_interpolation)
 * [Chroma key compositing](https://www.wikiwand.com/en/Chroma_key)
 * Detail/Edge enhancement
 * Noise reduction
 * Brightness/Contrast/Hue/Saturation/Sharpness/Gamma adjustments
