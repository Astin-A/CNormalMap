/***************************************************************************
 * C normal map generator
 *
 * Basic use:
*     #define C_NORMALMAP_IMPLEMENTATION before including this file to get
        the implementation. Otherwise this acts as a regualr header file
        

 *     uint32_t *in = ...load pixels from image
 *     uint32_t *nm = cinm_normal_map(in, w, h, scale, blurRadius, greyscaleType);
 *     ...write normal map to a file
 *
 *  Other defines you can use(before including this file):
 *  #define C_NORMALMAP_USE_CIMD for SSE/AVX
 *  #define C_NORMALMAP_NO_CIMD to ignore SSE/AVX functions
 *  #define C_NORMALMAP_STATIC for static defintions(no extern functions)
 ***************************************************************************/


#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef CINM_DEF
#ifdef C_NORMALMAP_STATIC
#define CINM_DEF static
#else
#define CINM_DEF extern
#endif
#endif

#ifndef _MSV_VER
    #ifdef __cplusplus
    #define cinm_inline inline
    #else
    #define cinm_inline
    #endif
#else
    #define cinm_inline __forceinline
#endif


#ifndef CINM_GREYSCALE_TYPE
#define CINM_GREYSCALE_TYPE

typedef enum
{
    cinm_greyscale_none,
    cinm_greyscale_lightness,
    cinm_greyscale_average,
    cinm_greyscale_luminance,
    cinm_greyscale_count,
} cinm_greyscale_type;
#endif

#ifndef C_NORMALMAP_IMPLEMENTATION

CINM_DEF void cinm_greyscale(const uint32_t *in, uint32_t *out, int32_t w, int32_t h, cinm_greyscale_type type);

//Converts values in "buffer" to greyscale  using either the
//lightness, average or luminance methods
//Result can be produced in-place if "in" and "out" are the same buffers


CINM_DEF uint32_t *cinm_normal_map(const uint32_t *in, int32_t w, int32_t h, float scale, float blurRadius, cinm_greyscale_type greyscaleType);
//Converts input buffer to a normal map and returns a pointer to it. 
//  "scale" controls the intensity of the result
//  "blurRadius" controls the radius for gaussian blurring before generating normals
//  "greyscaleType" specifies the conversion method from color to greyscale before 
//   generating the normal map. This step is skipped when using cinm_greyscale_none.

#else //C_NORMALMAP_IMPLEMENTATION

#ifdef C_NORMALMAP_NO_CIMD

#include <assert.h>
#include <intrin.h> 
#include <emmintrin.h> 

#ifdef __AVX__

#define CINM_CIMD_INCREMENT 8

#define cimd__int __m256i
#define cimd__float __m256
#define cimd__set1_epi32(a)     _mm256_set1_epi32(a)
#define cimd__setzero_ix()      _mm256_setzero_si256()
#define cimd__setzero_rx()      _mm256_setzero_ps()
#define cimd__and_ix(a, b)      _mm256_and_si256(a, b)
#define cimd__or_ix(a, b)       _mm256_or_si256(a, b)
#define cimd__add_epi32(a, b)   _mm256_add_epi32(a, b)
#define cimd__sub_epi32(a, b)   _mm256_sub_epi32(a, b)
#define cimd__max_epi32(a, b)   _mm256_max_epi32(a, b)
#define cimd__min_epi32(a, b)   _mm256_min_epi32(a, b)
#define cimd__loadu_ix(a)       _mm256_loadu_si256(a)
#define cimd__storeu_ix(ptr, v) _mm256_storeu_si256(ptr, v)
#define cimd__srli_epi32(a, i)  _mm256_srli_epi32(a, i)
#define cimd__slli_epi32(a, i)  _mm256_slli_epi32(a, i)
#define cimd__set1_ps(a)        _mm256_set1_ps(a)
#define cimd__cvtepi32_ps(a)    _mm256_cvtepi32_ps(a)
#define cimd__cvtps_epi32(a)    _mm256_cvtps_epi32(a)
#define cimd__add_ps(a, b)      _mm256_add_ps(a, b)
#define cimd__mul_ps(a, b)      _mm256_mul_ps(a, b)

#else


#define CINM_CIMD_INCREMENT 4
#define cimd__int __m128i
#define cimd__float __m128
#define cimd__set1_epi32(a)     _mm_set1_epi32(a)
#define cimd__setzero_ix()      _mm_setzero_si128()
#define cimd__setzero_rx()      _mm_setzero_ps()
#define cimd__and_ix(a, b)      _mm_and_si128(a, b)
#define cimd__or_ix(a, b)       _mm_or_si128(a, b)
#define cimd__add_epi32(a, b)   _mm_add_epi32(a, b)
#define cimd__sub_epi32(a, b)   _mm_sub_epi32(a, b)
#define cimd__max_epi32(a, b)   _mm_max_epi32(a, b)
#define cimd__min_epi32(a, b)   _mm_min_epi32(a, b)
#define cimd__loadu_ix(a)       _mm_loadu_si128(a)
#define cimd__storeu_ix(ptr, v) _mm_storeu_si128(ptr, v)
#define cimd__srli_epi32(a, i)  _mm_srli_epi32(a, i)
#define cimd__slli_epi32(a, i)  _mm_slli_epi32(a, i)
#define cimd__set1_ps(a)        _mm_set1_ps(a)
#define cimd__cvtepi32_ps(a)    _mm_cvtepi32_ps(a)
#define cimd__cvtps_epi32(a)    _mm_cvtps_epi32(a)
#define cimd__add_ps(a, b)      _mm_add_ps(a, b)
#define cimd__mul_ps(a, b)      _mm_mul_ps(a, b)

#endif //AVX_AVAILABLE
#endif //C_NORMALMAP_NO_CIMD

#define cinm__min(a, b) ((a) < (b) ? (a) : (b))
#define cinm__max(a, b) ((a) > (b) ? (a) : (b))

typedef struct 
{
    int32_t x, y;
} cinm__v2i;

typedef struct 
{
    float x,y,z;
} cinm__v3;


cinm_inline static float 
cinm__length(float x, float y, float z) 
{ 
    return sqrtf(x*x + y*y + z*z);
}


cinm_inline static cinm__v3 
cinm__normalized(float x, float y, float z) 
{
    cinm__v3 result;
    float len = cinm__length(x, y, z);
    if(len > 1e-04f) {
        float invLen = 1.0f / len;
        result.x = x*invLen;
        result.y = y*invLen;
        result.z = z*invLen;
    } else {
        result.x = result.y = result.z = 0.0f;
    }
    return result;
}

cinm_inline static uint32_t 
cinm__lightness_average(uint32_t r, uint32_t g, uint32_t b)
{
    return (cinm__max(cinm__max(r, g), b)+cinm__min(cinm__min(r, g), b))/2;
}

cinm_inline static uint32_t 
cinm__average(uint32_t r, uint32_t g, uint32_t b)
{
    return (r+g+b)/3;
}
//NOTE: bias is based on human eye sensitivity 
cinm_inline static uint32_t 
cinm__luminance(uint32_t r, uint32_t g, uint32_t b)
{
    return (uint32_t)(0.21f*r+0.72f*g+0.07f*b);
}

cinm_inline static uint32_t
cinm__greyscale_from_byte(uint8_t c)
{
    return (c | c << 8u | c << 16u | 255u << 24u);  
}

CINM_DEF void
cinm__generate_gaussian_box(float *outBoxes, int32_t n, float sigma)
{
    float wIdeal = sqrtf((12.0f*sigma*sigma/n)+1.0f);
    int32_t wl = floorf(wIdeal);
    if(wl%2 == 0) --wl;
    int32_t wu = wl+2;

    float mIdeal = (12.0f*sigma*sigma - n*wl*wl - 4.0f*n*wl - 3.0f*n)/(-4.0f*wl - 4.0f);
    int32_t m = roundf(mIdeal);

    for(int i = 0; i < n; ++i) {
        outBoxes[i] = (i < m) ? wl : wu;
    }
}

CINM_DEF void 
cinm__box_blur_h(uint32_t *in, uint32_t *out, int32_t w, int32_t h, float r)
{
    float invR = 1.0f/(r+r+1);
    for(int i = 0; i < h; ++i) {
        int32_t oi = i*w;
        int32_t li = oi;
        int32_t ri = oi+r;

        uint32_t fv  = in[oi] & 0xFFu;
        uint32_t lv  = in[oi+w-1] & 0xFFu;
        uint32_t sum = (r+1)*fv;

        for(int j = 0; j < r; ++j) {
            sum += in[oi+j] & 0xFFu;
        }
        for(int j = 0; j <= r; ++j) {
            sum += (in[ri++] & 0xFFu) - fv;
            out[oi++] = cinm__greyscale_from_byte((uint8_t)(sum*invR));
        }
        for(int j = r+1; j < w-r; ++j) {
            sum += (in[ri++] & 0xFFu) - (in[li++] & 0xFFu);
            out[oi++] = cinm__greyscale_from_byte((uint8_t)(sum*invR));
        }
        for(int j = w-r; j < w; ++j) {
            sum += lv - (in[li++] & 0xFFu);
            out[oi++] = cinm__greyscale_from_byte((uint8_t)(sum*invR));
        }
    }
}

CINM_DEF void 
cinm__box_blur_v(uint32_t *in, uint32_t *out, int32_t w, int32_t h, float r)
{
    float invR = 1.0f/(r+r+1);
    for(int i = 0; i < w; ++i) {
        int32_t oi = i;
        int32_t li = oi;
        int32_t ri = oi+r*w;

        uint32_t fv  = in[oi] & 0xFFu;
        uint32_t lv  = in[oi+w*(h-1)] & 0xFFu;
        uint32_t sum = (r+1)*fv;

        for(int j = 0; j < r; j++) {
            sum += in[oi+j*w] & 0xFFu;
        }
        for(int j = 0; j <= r; j++) {
            sum += (in[ri] & 0xFFu) - fv;
            out[oi] = cinm__greyscale_from_byte((uint8_t)(sum*invR));
            ri+=w; oi+=w;
        }
        for(int j = r+1; j < h-r; j++) {
            sum += (in[ri] & 0xFFu) - (in[li] & 0xFFu);
            out[oi] = cinm__greyscale_from_byte((uint8_t)(sum*invR));
            li += w; ri+=w; oi+=w;
        }
        for(int j = h-r; j < h; j++) {
            sum += lv - (in[li] & 0xFFu);
            out[oi] = cinm__greyscale_from_byte((uint8_t)(sum*invR));
            li += w; oi+=w;
        }
    }
}

CINM_DEF void 
cinm__gaussian_box(uint32_t *in, uint32_t *out, int32_t w, int32_t h, float r)
{
    float boxes[3];
    cinm__generate_gaussian_box(boxes, sizeof(boxes)/sizeof(boxes[0]), r);

    cinm__box_blur_h(in, out, w, h, (boxes[0]-1)/2);
    cinm__box_blur_v(out, in, w, h, (boxes[0]-1)/2);
    cinm__box_blur_h(in, out, w, h, (boxes[1]-1)/2);
    cinm__box_blur_v(out, in, w, h, (boxes[1]-1)/2);
    cinm__box_blur_h(in, out, w, h, (boxes[2]-1)/2);
    cinm__box_blur_v(out, in, w, h, (boxes[2]-1)/2);

    memcpy(out, in, w*h*sizeof(uint32_t));
}


CINM_DEF void 
cinm__sobel3x3_normals(const uint32_t *in, uint32_t *out, int32_t w, int32_t h, float scale)
{
    const float xk[3][3] = {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1},
    };

    const float yk[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1},
    };

    //TODO: optimize
    for(int32_t y = 0; y < h; ++y) {
        for(int32_t x = 0; x < w; ++x) {
            float xmag = 0.0f;
            float ymag = 0.0f;

            for(int32_t a = 0; a < 3; ++a) {
                for(int32_t b = 0; b < 3; ++b) {
                    int32_t xIdx = cinm__min(w-1, cinm__max(1, x+b-1));
                    int32_t yIdx = cinm__min(h-1, cinm__max(1, y+a-1));
                    int32_t index = yIdx*w+xIdx;
                    uint32_t pixel = in[index] & 0xFFu;
                    xmag += pixel*xk[a][b];
                    ymag += pixel*yk[a][b];
                }
            }

            cinm__v3 color = cinm__normalized(xmag*scale, ymag*scale, 255.0f);
            uint32_t r = (uint32_t)((1.0f+color.x)*0.5f*255);
            uint32_t g = (uint32_t)((1.0f+color.y)*0.5f*255);
            uint32_t b = (uint32_t)((1.0f+color.z)*0.5f*255);
            out[y*w+x] = (r | g << 8u | b << 16u | 255 << 24u);
        }
    }
}
static void
cinm__greyscale(const uint32_t *in, uint32_t *out, int32_t w, int32_t h, cinm_greyscale_type type)
{
    switch(type) {
        case cinm_greyscale_lightness : {
            for(int32_t i = 0; i < count; ++i) {
                uint32_t c = in[i];
                uint32_t l = cinm__lightness_average(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu);
                out[i] = cinm__greyscale_from_byte(l);
            }
        } break;

        case cinm_greyscale_average : {
            for(int32_t i = 0; i < count; ++i) {
                uint32_t c = in[i];
                uint32_t l = cinm__average(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu);
                out[i] = cinm__greyscale_from_byte(l);
            }
        } break;

        case cinm_greyscale_luminance : {
            for(int32_t i = 0; i < count; ++i) {
                uint32_t c = in[i];
                uint32_t l = cinm__luminance(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu);
                out[i] = cinm__greyscale_from_byte(l);
            }
        } break;
    }
}

#ifndef C_NORMALMAP_NO_CIMD
static void
cinm__cimd_greyscale(const uint32_t *in, uint32_t *out, int32_t w, int32_t h, cinm_greyscale_type type)
{
    cimd__int redMask   = cimd__set1_epi32(0xFF);
    cimd__int greenMask = cimd__set1_epi32(0xFF00u);
    cimd__int blueMask  = cimd__set1_epi32(0xFF0000u);
    cimd__int alpha     = cimd__set1_epi32(0xFF000000u);


    int32_t count = w*h;

    switch(type) {
        case cinm_greyscale_lightness : {
            for(int32_t i = 0; i < count; i += CINM_CIMD_INCREMENT) {
                cimd__int c = cimd__loadu_ix((cimd__int *)&in[i]);
                cimd__int r = cimd__and_ix(c, redMask);
                cimd__int g = cimd__srli_epi32(cimd__and_ix(c, greenMask), 8);
                cimd__int b = cimd__srli_epi32(cimd__and_ix(c, blueMask), 16);

                cimd__int max = cimd__max_epi32(cimd__max_epi32(r, g), b);
                cimd__int min = cimd__min_epi32(cimd__min_epi32(r, g), b);
                cimd__int l   = cimd__srli_epi32(cimd__add_epi32(min, max), 1);

                l = cimd__or_ix(cimd__slli_epi32(l, 16), 
                    cimd__or_ix(cimd__slli_epi32(l, 8), 
                    cimd__or_ix(l, alpha)));

                cimd__storeu_ix((cimd__int *)&out[i], l);
            }
        } break;

        case cinm_greyscale_average : {
            cimd__float inverse3 = cimd__set1_ps(1.0f/3.0f);
            for(int32_t i = 0; i < count; i += CINM_CIMD_INCREMENT) {
                cimd__int c = cimd__loadu_ix((cimd__int *)&in[i]);
                cimd__int r = cimd__and_ix(c, redMask);
                cimd__int g = cimd__srli_epi32(cimd__and_ix(c, greenMask), 8);
                cimd__int b = cimd__srli_epi32(cimd__and_ix(c, blueMask), 16);

                //NOTE: integer division is only available in SVML(not sse or avx) which I'm 
                //not going to use. Agner Fog has an efficient division implementation but I am 
                //simply going to use an inverse multiply in float and convert back to int. 
                //This may be changed later
                cimd__int s = cimd__add_epi32(cimd__add_epi32(r, g), b);
                s = cimd__cvtps_epi32(cimd__mul_ps(cimd__cvtepi32_ps(s), inverse3));
                s = cimd__or_ix(cimd__slli_epi32(s, 16), 
                    cimd__or_ix(cimd__slli_epi32(s, 8), 
                    cimd__or_ix(s, alpha)));

                cimd__storeu_ix((cimd__int *)&out[i], s);
            }
        } break;

        case cinm_greyscale_luminance : {
            cimd__float rBias = cimd__set1_ps(0.21f);
            cimd__float gBias = cimd__set1_ps(0.72f);
            cimd__float bBias = cimd__set1_ps(0.07f);

            for(int32_t i = 0; i < count; i += CINM_CIMD_INCREMENT) {
                cimd__int c = cimd__loadu_ix((cimd__int *)&in[i]);
                cimd__float r = cimd__cvtepi32_ps(cimd__and_ix(c, redMask));
                cimd__float g = cimd__cvtepi32_ps(cimd__srli_epi32(cimd__and_ix(c, greenMask), 8));
                cimd__float b = cimd__cvtepi32_ps(cimd__srli_epi32(cimd__and_ix(c, blueMask), 16));

                r = cimd__mul_ps(r, rBias);
                g = cimd__mul_ps(g, gBias); 
                b = cimd__mul_ps(b, bBias);

                cimd__int sum = cimd__cvtps_epi32(cimd__add_ps(r, cimd__add_ps(g, b)));

                sum = cimd__or_ix(cimd__slli_epi32(sum, 16), 
                      cimd__or_ix(cimd__slli_epi32(sum, 8), 
                      cimd__or_ix(sum, alpha)));

                cimd__storeu_ix((cimd__int *)&out[i], sum);
            }
        } break;
    }
}
#endif //C_NORMALMAP_NO_CIMD


CINM_DEF void
cinm_greyscale(const uint32_t *in, uint32_t *out, int32_t w, int32_t h, cinm_greyscale_type type)
{

#ifndef C_NORMALMAP_NO_CIMD
    int32_t count = w*h;
    if(count > 0 && count % CINM_CIMD_INCREMENT == 0) { 
        cinm__cimd_greyscale(in, out, w, h, type);
    } else {
        cinm__greyscale(in, out, w, h, type);
    }
#else
    cinm__greyscale(in, out, w, h, type);
#endif

}

CINM_DEF uint32_t * 
cinm_normal_map(const uint32_t *in, int32_t w, int32_t h, float scale, float blurRadius, cinm_greyscale_type greyscaleType)
{
    uint32_t *intermediate = (uint32_t *)malloc(w*h*sizeof(uint32_t));
    if(!intermediate) return NULL;

    uint32_t *result = malloc(w*h*sizeof(uint32_t));
    if(result) {
        if(greyscaleType != cinm_greyscale_none) {
            cinm_greyscale(in, result, w, h, greyscaleType);
        } else {
            memcpy(result, in, w*h*sizeof(uint32_t));
        }

        float radius = cinm__min(cinm__min(w,h), cinm__max(0, blurRadius));
        cinm__gaussian_box(result, intermediate, w, h, radius);
        cinm__sobel3x3_normals(intermediate, result, w, h, scale);
    }

    free(intermediate);
    return result;
}

#endif //ifndef C_NORMALMAP_IMPLEMENTATION
