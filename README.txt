==================================================
         SCIKIT-IMAGE UPSCALING LIBRARY
==================================================
             (c) Philip Blair 2015
             Licensed under LGPLv3
==================================================
Description
------------
This library implements a few image upscaling
algorithms which are designed to interoperate
with images as represented by the scikit-image
library. It provides the following:

  - Nearest Neighbor Interpolation
  - Bilinear Interpolation
  - Bicubic Interpolation
  - Directional Cubic Convolution Interpolation

This package was made for a class project, so
it is nothing too major; however, I am publishing
it in case someone finds it useful. Want to add
something? Feel free to open a pull request!
Something broken? Open an issue!

-------------------------------------------------
Installation
-------------
As of now, the recommended way of installing
this package is using git and pip:

  1. Clone this repository and navigate to it
     in your shell of choice
  2. Inside of this folder, run `pip install ./`
  
-------------------------------------------------
Usage
------
The library exports the following classes:

  - NearestNeighbor
  - Bilinear
  - Bicubic
  - DCCI

Upscaling is performed via the static
method `.upscale(<img>, <params>...)`.
For now, the `<params>...` argument is
the same single argument for all algorithms:
the amount by which you would like to scale
the image (e.g. 2 will double the image size).
Here is an example:

  > # img is some image that's been loaded
  > from skupscale import Bilinear
  > upscaled = Bilinear.upscale(img,2)

And that's really all there is to it!
