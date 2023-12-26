# CNormalMap
 A very simple Single Header normal map generator library


 ## WORK IN PROGRESS. USE AT OWN RISK

A very simple normal map generator written as a single header library. 

Features:
 - Convert color buffer to greyscale using either lightness, average or luminance methods(with SSE/AVX versions available)
 - Convert greyscale buffer to a normal map

Possible features to add
 - SIMD version of normal map generation
 - Filtering options for normal map generation (ie: 3x3 sobel, etc)
 - Blurring functionality. But most likely this will be made in a seperate library. I'd prefer to keep this simple.

Basic example:
```C
#define C_NORMALMAP_IMPLEMENTATION
#include "CNormalMap.h"

...

uint32_t *image = ...load pixels from some image;
cinm_greyscale(image, imageWidth*imageHeight, cinm_greyscaleType_average);

uint32_t *normalmap = ...allocate a second buffer of the same dimensions
cinm_normal_map(image, normalmap, imageWidth, imageHeight, 1.0f);

... write normalmap to a file

```
