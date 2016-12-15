GPU-Accelerated Obstruction Free Photography

## Milestone 1
[Presentation](https://docs.google.com/presentation/d/1DC9_Wc-EFBK_Mkw0OPGu9UvurDkdzBU-ETGmCi3bOsI/edit?usp=sharing)

## Milestone 2

## Milestone 3

# Overview
The field of computer vision is the dual to computer graphics. In many cases, the unknowns that we are solving for in optical models are swapped around. For example, the graphics problem "given camera motion and lighting + scene conditions, generate a photorealistic image" has the counterpart vision problem "given photorealistic images and estimated lighting + scene conditions, calculate the camera motion".

In this project, we take on the challenge of recovering a background scene from an image with both a background scene and an occluding layer (e.g. fence, glass pane reflection, non-symmetric objects). The input to the algorithm is a series of images extracted from a camera that pans across a scene in planar motion. The inspiration and intuition driving the solution of this challenge involves leveraging parallax motion of the foreground and background images to separate a foreground and background layer. Given the data parallel property of the challenge (per-pixel calculation), this algorithm is a perfect target for GPU acceleration. While the main focus is emphasized on the performance gained by implementing the algorithm on the GPU (as opposed to the CPU), we will briefly go over the algorithm.

# Algorithm Outline
Let's briefly go over the algorithm to get an idea of what's going on under the hood.

## Description
Input: a sequence of images (we use 5) with the middle one being the "reference frame"

Assumptions:
 * There needs to be a distinct occluding layer and background layer
 * The majority of the background and foreground motion needs to be uniform
 * Every background pixel is visible in at least one frame

Output: A background image without obstruction

## Pipeline

1) Initialization
  * Edge Detection
  * Sparse flow
  * Layer separation
  * Dense flow interpolation
  * Warm matrix generation + Image warping
  * Background and foreground initial estimated separation
2) Optimization
  * Minimize cost function

# Implementation
Here we describe how we implement each step in the pipeline.

## Edge Detection
We use Canny's edge detection algorithm that has the sub-pipeline as follows:

1) Grayscale conversion
2) Noise removal (smoothing)
3) Gradient calculation
4) Non-maximum edge suppression
5) Hysteresis




# Results

## Performance Analysis

## Time Breakdown

# Future work

# Bloopers

# Acknowledgements
