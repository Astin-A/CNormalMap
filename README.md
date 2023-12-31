# CNormalMap
A very simple normal map generator written as a single header library.


Features:
 - Convert color buffer to greyscale using either lightness, average or luminance methods(with SSE/AVX versions available)
 - Convert greyscale buffer to a normal map with gaussian blur pre-filtering

### example:

Input:

![input](https://imgur.com/Grx9Uvs.png) 

Output:

![output](https://i.imgur.com/m64imlB.png)

Comparison of lighting without and with normal map(and a second "detail" normal map)

![lighting](https://imgur.com/CIw2oFB.png)


### Interface
```C
typedef enum
{
    cinm_greyscale_none,
    cinm_greyscale_lightness,
    cinm_greyscale_average,
    cinm_greyscale_luminance,
    cinm_greyscale_count,
} cinm_greyscale_type;

void cinm_greyscale(const uint32_t *in, uint32_t *out, int32_t w, int32_t h, cinm_greyscale_type type);

uint32_t *cinm_normal_map(const uint32_t *in, int32_t w, int32_t h, float scale, float blurRadius, cinm_greyscale_type greyscaleType);
```

### Basic Usage:
```C
#define C_NORMALMAP_IMPLEMENTATION
#include "CNormalMap.h"

...

uint32_t *image = ...load pixels from some image;
uint32_t *nm = cinm_normal_map(image, imageWidth, imageHeight, 1.0f, 2.0f, cinm_greyscale_average); 


... write normalmap to a file

```


