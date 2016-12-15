GPU-Accelerated Obstruction Free Photography

Names: [David Liao](https://github.com/davlia) and [Zhan Xiong Chin](https://github.com/czxcjx)

Tested on AWS g2.2xlarge instance (Linux ip-172-31-53-81 4.4.0-53-generic #74-Ubuntu SMP Fri Dec 2 15:59:10 UTC 2016 x86_64 x86_64 x86_64 GNU/Linux, Nvidia GK104GL [GRID 520])

## Milestone 1
[Presentation](https://docs.google.com/presentation/d/1DC9_Wc-EFBK_Mkw0OPGu9UvurDkdzBU-ETGmCi3bOsI/edit?usp=sharing)

## Milestone 2
[Presentation]()

## Milestone 3
[Presentation]()

## Final Presentation
[Presentation](https://docs.google.com/presentation/d/1GOTx-BjeLcm1W14b2YrThjsyaG2mvMA-fIdKSFwgc-E/edit?usp=sharing)

<img src="https://i.imgur.com/qJYtNG1.png" width=300></img> <img src="https://i.imgur.com/Rac9baV.png" width=300></img>

# Build instructions
Just run make. Requires CUDA, OpenCV, and CImg.

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

1) [Initialization](#edge-detection)
  * [Edge detection](#edge-detection)
  * [Sparse flow](#sparse-flow)
  * [Layer separation](#layer-separation)
  * [Dense flow interpolation](#dense-flow-interpolation)
  * [Warp matrix generation + Image warping](#warp-matrix)
  * [Background and foreground initial estimated separation](#background-and-foreground-initial-estimation)
2) [Optimization](#optimization)
  * [Minimize cost function](#optimization)

# Implementation
Here we describe how we implement each step in the pipeline.

## Edge Detection

![](https://i.imgur.com/MR0CNhl.png)

We use Canny's edge detection algorithm that has the sub-pipeline as follows:

1) Grayscale conversion
2) Noise removal (smoothing)
3) Gradient calculation
4) Non-maximum edge suppression
5) Hysteresis

Step 1 is an intensity average and steps 2 + 3 are convolutions using a Gaussian kernel and Sobel operators.

Step 4 takes the edge direction and inspects pixels on both sides of the edge (in the forward and reverse direction of the gradient) and if the pixel is not a "mountain", it will be suppressed. The end result is thinned edges that are pixel thick.

Step 5 is a technique to reduce noise in the edges when thresholding. When setting an intensity, threshold, it is possible that an edge dances around the boundaries of the threshold. The result is a noisy dotted edge that we don't want. Instead, we set a lower threshold and upper threshold where if an edge starts above the upper threshold but dips below the upper threshold while staying above the lower threshold, it will still be considered an edge. The below image borrowed from OpenCV docs gives a good idea of what this step does.
![](http://docs.opencv.org/3.1.0/hysteresis.jpg)
Here the vertical represents the intensity while the horizontal maps to some 1D projected plane of the edge. Line A is considered an edge even though C is below the threshold. Line B, however, is not since it begins, ends and maintains within the two thresholds.

## Sparse Flow
We then need to calculate how the image moves. The original paper defines a Markov Random Field over the edges, constraining pixels to match the location they flow to, as well as being similar to the flow of their neighboring pixels. We were not able to replicate this method, instead opting to use a more common optical flow algorithm, the [Lucas-Kanade method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method), to approximate this. We made use of the OpenCV implementation for this step. 

We run the Lucas-Kanade algorithm on the set of pixels that make up the edges. For each pixel, the Lucas-Kanade method assumes that the displacement of a small window (e.g. 21x21 pixels) around the window is constant between two frames, and sets up a system of linear equations on the spatial and temporal gradients of the window of pixels. Then, it uses least squares to solve this system of equations, with the result being the optical flow of the chosen pixel. While this is also ideal for GPU acceleration, as each pixel can be treated independently, we did not have time to write a kernel for this step. Also, the method no longer maintains similarity between adjacent pixels, so the flow may not be as continuous as the original paper's method.

## Layer Separation
![](https://i.imgur.com/b7SXYN8.png)


To then figure out which edges are part of the occluding layer and which edges are part of the background layer, we use RANSAC to separate the vertices into two clusters.

For this, we make the assumption that both the occluding layer and the background layer are translated in the same way between two frames. This is also a simplification from the original paper's method, which estimates a homography rather than a simple translation.
 
Then, we fix a percentage P% of pixels that will be classified as part of the background layer. We pick a random initial starting vector v, then take the nearest P% of motion vectors to it and classify that as our background layer. Then, we take the mean of this newly computed background layer's optical flow, and treat this as our new vector v. Repeating the process a fixed number of iterations, we end up with a separation of pixels into background and foreground layer.

This did not always give stable results, so we augmented this with additional heuristics, such as separating based on the intensity of the pixels (e.g. when the occluding layer is a fence or racket, "whiter" areas tend to be part of the occluding layer, showing wall or sky)

## Dense Flow Interpolation
After we figure out how the edges move and separate the, we interpolate sparse motion vectors of the background layer into a dense motion field. Basically we take the edge motion vectors and interpolate to figure out how every other pixel moves. There were two methods we used to calculate this. One was to do a K-nearest neighbors weighted interpolation, while the other was a naive average to generate an affine transformation matrix. As it turns out, the latter approximation works fine in the context of an initial estimate.

## Warp Matrix
We take the background motion field from the last step and generate a warp matrix (basically inverting the motion matrix). Then apply it to all the non-reference images to warp them to the reference image. This has the effect of moving moving all the images in a way such that their backgrounds are stacked on top of each other. Their foreground, as a result, is the only moving layer. 

## Background and Foreground Initial Estimation
After the last step, we've reduced the problem to a state where the camera perspective is static. We can then estimate a pixel for each layer based on this information. 

![](https://i.imgur.com/dOO715o.png)

Mean based estimation


If the occluding the layer is opaque, we can take the average of all the pixels at a location. If it is reflective, we can take the minimum (since reflection is additive light). 


Alternatively, we can employ smarter techniques that take into account spatial coherence. In this case, we will select the pixel that has the most number of neightbors with a pixel intensity similar to it. 

<img src="https://i.imgur.com/sgjj7ZN.png" width=550></img>

Spatial coherence based estimation

## Optimization
The quality of the image can be measured by an objective function defined over the foreground and background layers and motion fields, as well as the alpha mask between the two layers. Then, optimizing this objective function will (in principle) give a better image.

The original paper made use of an objective function that penalizes the following terms:

1. Differences between the original image and the combination of the estimated background and foreground
2. Noisy alpha mask
3. Noisy background/foreground images
4. High gradients on the same pixel of the background and foreground image (i.e. highly detailed of the image probably only belong to one layer rather than a mix of both)
5. Noisy motion fields

They optimized this using iteratively reweighted least squares, using the pyramid method to start from a coarse image and optimize on successively finer ones. 

For our code, we made use of the same objective function, but we performed a naive gradient descent to optimize the objective function. Tweaking the parameters of the optimization, we sometimes were able to obtain slightly better results, but in general, we did not find a significant improvement in image quality, possibly due to the slow convergence of our naive gradient descent approach. 


# Results
The pipeline benchmarked exclusively for the individual kernel invocations. The overhead costs were excluded in this analysis as they are not useful/relevant. The input used was `img/hanoiX.png` where X is replaced by the image number. **Please look at the entire recorded data set [here](https://docs.google.com/spreadsheets/d/1-figxG5rl99xXkUG_dW7jFhtrUXwWFpR-utSPinnJUk/edit?usp=sharing).** Note the tabs at the bottom!

## Performance Analysis
![](http://i.imgur.com/ueHztLu.png)
We note some of the more important numbers here. At a glance, it's pretty obvious that the pipeline hugely benefited from GPU acceleration. The most notable is the gradient descent optimization step which received a huge speedup. Edge detection, generating foreground and background estimates, calculating spatial coherence, and gradient descent are all low-cost computations (\<5ms). This makes sense as most of the work that they do is very computational. They all have low branch counts and perform only the most basic of logical evaluations. 

Since the pipeline is sequential, performance bottlenecks are distributed across each pipeline step and can be targets for future improvement. The most notable bottleneck is the RANSAC algorithm. Since it is an iterative model, tweaking iteration parameters would speed it up at the cost of accuracy. At its current benchmarked state, it is running ~50 iterations. So each iteration runs at about ~2 ms. 

## Time Breakdown
![](http://i.imgur.com/Jtg5HbM.png)
The overall time spent computing on the GPU was on RANSAC as well as Gradient descent. This is because gradient descent and RANSAC are both iterative methods that require multiple steps to convergence. As a result, we want high iterations for for both of these to get the best images. On the other hand, if we were to make this an real-time pipeline, we would choose to do ~5 iterations to achieve sub 100ms time. 

# Other images
![](https://i.imgur.com/cBbQCz2.png)

Reflection removal as well. Top left: original image, top right: separation, bottom left: without gradient descent, bottom right: with gradient descent

![](https://i.imgur.com/pnbd2QE.png)

Removing the spokes of a bike wheel from Harrison College House.

# Bloopers
<img src="http://i.imgur.com/2MHWLro.png" width=600></img>

When canny edge detection goes wrong...

<img src="http://i.imgur.com/Rs6V7kB.png" width=600></img>

A bug in the pipeline propagates and magnifies in the pipeline downstream.

![](http://i.imgur.com/HO0umiJ.png)
Maya scene where we have grounded camera movements as well as distances. But still wrong output!

# Acknowledgements
We want to give a big thanks to Patrick Cozzi for teaching the CIS 565 course as well as Benedict Brown for giving us general CV tips.
