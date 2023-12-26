/***************************************************************************
 * C normal map generator
 *
 * Basic use:
 *     ...load your pixels to a buffer and convert to greyscale
 *     cinm_greyscale(inBuffer, pixelCount, greyscaleConversionType);
 *
 *     ...make an output buffer for normal map gen from greyscale buffer
 *     cinm_normal_map(inBuffer, outBuffer, w, h, scale);
 *
 *     ...write output buffer to a file
 ***************************************************************************/


#include <stdint.h>

#ifndef CINM_DEF
#ifdef CINM_STATIC
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

#ifndef SI_NORMALMAP_IMPLEMENTATION
enum cinm_greyscale_type;

//Converts values in "buffer" to greyscale  using either the
//lightness, average or luminance methods
CINM_DEF void cinm_greyscale(uint32_t *buffer, int32_t count, greyscale_type type);

//Converts input buffer to a normal map and writes it to the output buffer
//Colors should be converted to greyscale for proper results
//"scale" adjusts the intensity of the result
CINM_DEF void cinm_normal_map(const uint32_t *inBuffer, uint32_t *outBuffer, int32_t w, int32_t h, int32_t comp, float scale);

#else //SI_NORMALMAP_IMPLEMENTATION

#ifdef SI_NORMALMAP_USE_SIMD

#include <intrin.h> 
#include <emmintrin.h> 

#ifdef __AVX__
#define CINM_SIMD_INCREMENT 8

#define cimd__int __m256i
#define cimd__float __m256

#define cimd__set1_epi32(a) _mm256_set1_epi32(a)
#define cimd__setzero_ix()  _mm256_setzero_si256()
#define cimd__setzero_rx()  _mm256_setzero_ps()

#define cimd__and_ix(a, b) _mm256_and_si256(a, b)
#define cimd__or_ix(a, b)  _mm256_or_si256(a, b)

#define cimd__add_epi32(a, b) _mm256_add_epi32(a, b)
#define cimd__sub_epi32(a, b) _mm256_sub_epi32(a, b)

#define cimd__max_epi32(a, b) _mm256_max_epi32(a, b)
#define cimd__min_epi32(a, b) _mm256_min_epi32(a, b)

#define cimd__loadu_ix(a)  _mm256_loadu_si256(a)
#define cimd__storeu_ix(ptr, v) _mm256_storeu_si256(ptr, v)

#define cimd__srli_epi32(a, i) _mm256_srli_epi32(a, i)
#define cimd__slli_epi32(a, i) _mm256_slli_epi32(a, i)

#define cimd__set1_ps(a) _mm256_set1_ps(a)
#define cimd__cvtepi32_ps(a) _mm256_cvtepi32_ps(a)
#define cimd__cvtps_epi32(a) _mm256_cvtps_epi32(a)

#define cimd__add_ps(a, b) _mm256_add_ps(a, b)
#define cimd__mul_ps(a, b) _mm256_mul_ps(a, b)

#else

#define cimd__int __m128i
#define cimd__float __m128

#define CINM_SIMD_INCREMENT 4
#define cimd__set1_epi32(a) _mm_set1_epi32(a)
#define cimd__setzero_ix()  _mm_setzero_si128()
#define cimd__setzero_rx()  _mm_setzero_ps()

#define cimd__and_ix(a, b) _mm_and_si128(a, b)
#define cimd__or_ix(a, b)  _mm_or_si128(a, b)

#define cimd__add_epi32(a, b) _mm_add_epi32(a, b)
#define cimd__sub_epi32(a, b) _mm_sub_epi32(a, b)

#define cimd__max_epi32(a, b) _mm_max_epi32(a, b)
#define cimd__min_epi32(a, b) _mm_min_epi32(a, b)

#define cimd__loadu_ix(a)  _mm_loadu_si128(a)
#define cimd__storeu_ix(ptr, v) _mm_storeu_si128(ptr, v)

#define cimd__srli_epi32(a, i) _mm_srli_epi32(a, i)
#define cimd__slli_epi32(a, i) _mm_slli_epi32(a, i)

#define cimd__set1_ps(a) _mm_set1_ps(a)
#define cimd__cvtepi32_ps(a) _mm_cvtepi32_ps(a)
#define cimd__cvtps_epi32(a) _mm_cvtps_epi32(a)

#define cimd__add_ps(a, b) _mm_add_ps(a, b)
#define cimd__mul_ps(a, b) _mm_mul_ps(a, b)

#endif //AVX_AVAILABLE
#endif //SI_NORMALMAP_USE_SIMD

typedef enum
{
    cinm_greyscaleType_lightness,
    cinm_greyscaleType_average,
    cinm_greyscaleType_luminance,
} cinm_greyscale_type;

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

cinm_inline static float 
cinm__linearize_srgb(float value)
{
    return value*value;
}

cinm_inline static cinm__v3 
cinm__normalized(float x, float y, float z) 
{
    sinm__v3 result;
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
    #define cinm__min(a, b) (((a) < (b)) ? (a) : (b))
    #define cinm__max(a, b) (((a) > (b)) ? (a) : (b))
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

static void
cinm__greyscale(uint32_t *buffer, int32_t count, cinm_greyscale_type type)
{
    switch(type) {
        case cinm_greyscaleType_lightness : {
            for(int32_t i = 0; i < count; ++i) {
                uint32_t c = buffer[i];
                uint32_t l = cinm__lightness_average(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu);
                buffer[i] = (255 << 24 | l << 16 | l << 8 | l);
            }
        } break;

        case cinm_greyscaleType_average : {
            for(int32_t i = 0; i < count; ++i) {
                uint32_t c = buffer[i];
                uint32_t l = cinm__average(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu);
                buffer[i] = (255 << 24 | l << 16 | l << 8 | l);
            }
        } break;

        case cinm_greyscaleType_luminance : {
            for(int32_t i = 0; i < count; ++i) {
                uint32_t c = buffer[i];
                uint32_t l = cinm__luminance(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu);
                buffer[i] = (255 << 24 | l << 16 | l << 8 | l);
            }
        } break;
    }
}

#ifdef SI_NORMALMAP_USE_SIMD
static void
cinm__cimd_greyscale(uint32_t *buffer, int32_t count, cinm_greyscale_type type)
{
    cimd__int redMask   = cimd__set1_epi32(0xFF);
    cimd__int greenMask = cimd__set1_epi32(0xFF00u);
    cimd__int blueMask  = cimd__set1_epi32(0xFF0000u);
    cimd__int alpha     = cimd__set1_epi32(0xFF000000u);

    switch(type) {
        case cinm_greyscaleType_lightness : {
            for(int32_t i = 0; i < count; i += CINM_SIMD_INCREMENT) {
                cimd__int c = cimd__loadu_ix((cimd__int *)&buffer[i]);
                cimd__int r = cimd__and_ix(c, redMask);
                cimd__int g = cimd__srli_epi32(cimd__and_ix(c, greenMask), 8);
                cimd__int b = cimd__srli_epi32(cimd__and_ix(c, blueMask), 16);

                cimd__int max = cimd__max_epi32(cimd__max_epi32(r, g), b);
                cimd__int min = cimd__min_epi32(cimd__min_epi32(r, g), b);
                cimd__int l   = cimd__srli_epi32(cimd__add_epi32(min, max), 1);

                l = cimd__or_ix(cimd__slli_epi32(l, 16), cimd__or_ix(cimd__slli_epi32(l, 8), cimd__or_ix(l, alpha)));

                cimd__storeu_ix((cimd__int *)&buffer[i], l);
            }
        } break;

        case cinm_greyscaleType_average : {
            cimd__float inverse3 = cimd__set1_ps(1.0f/3.0f);
            for(int32_t i = 0; i < count; i += CINM_SIMD_INCREMENT) {
                cimd__int c = cimd__loadu_ix((cimd__int *)&buffer[i]);
                cimd__int r = cimd__and_ix(c, redMask);
                cimd__int g = cimd__srli_epi32(cimd__and_ix(c, greenMask), 8);
                cimd__int b = cimd__srli_epi32(cimd__and_ix(c, blueMask), 16);

                //NOTE: integer division is only available in SVML(not sse or avx) which I'm 
                //not going to use. Agner Fog has an efficient division implementation but I am 
                //simply going to use an inverse multiply in float and convert back to int. 
                //This may be changed later
                cimd__int a = cimd__add_epi32(cimd__add_epi32(r, g), b);
                a = cimd__cvtps_epi32(cimd__mul_ps(cimd__cvtepi32_ps(a), inverse3));
                cimd__storeu_ix((cimd__int *)&buffer[i], a);
            }
        } break;

        case cinm_greyscaleType_luminance : {
            cimd__float rBias = cimd__set1_ps(0.21f);
            cimd__float gBias = cimd__set1_ps(0.72f);
            cimd__float bBias = cimd__set1_ps(0.07f);

            for(int32_t i = 0; i < count; i += CINM_SIMD_INCREMENT) {
                cimd__int c = cimd__loadu_ix((cimd__int *)&buffer[i]);
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

                cimd__storeu_ix((cimd__int *)&buffer[i], sum);
            }
        } break;
    }
}
#endif //SI_NORMALMAP_USE_SIMD

CINM_DEF void
cinm_greyscale(uint32_t *buffer, int32_t count, cinm_greyscale_type type)
{
#ifdef SI_NORMALMAP_USE_SIMD
    cinm__cimd_greyscale(buffer, count, type);
#else
    cinm__greyscale(buffer, count, type);
#endif
}

CINM_DEF void 
cinm_normal_map(const uint32_t *inBuffer, uint32_t *outBuffer, int32_t w, int32_t h, float scale)
{
    const float to01space = 1.0f / 255.0f;

    for(int32_t y = 0; y < h; ++y) {
        for(int32_t x = 0; x < w; ++x) {

            int32_t index = y * w + x;
            uint32_t curP = inBuffer[index];

            //NOTE: If curP is at an edge then just use curP as the "adjacent" pixel.
            //This gives us a clean edge as opposed to interpolating to some default value.
            uint32_t prevP  = (x > 0)      ? inBuffer[index-1] : curP;
            uint32_t aboveP = (y > 0)      ? inBuffer[index-w] : curP;
            uint32_t nextP  = (x != (w-1)) ? inBuffer[index+1] : curP;
            uint32_t underP = (y != (h-1)) ? inBuffer[index+w] : curP;

            //TODO: Support other channels? or just convert input to the expected format(using alpha only)?
            float h0 = (curP   & 0xFFu)*to01space;
            float h1 = (nextP  & 0xFFu)*to01space;
            float h2 = (underP & 0xFFu)*to01space;
            float h3 = (aboveP & 0xFFu)*to01space;
            float h4 = (prevP  & 0xFFu)*to01space;

            //NOTE: Interpolate half-way between adjacent height values
            //for a "ramp" effect between pixels as opposed to a "flat wall"
            //if we use exact pixel values.
            float rDiff = ((h0-h1)+(h4-h0))*0.5f;
            float gDiff = ((h0-h2)+(h3-h0))*0.5f;

            //NOTE: Scale here will increase/decrease the angle between the x/y axis and the z axis.
            //In other words(for x axis) for dot(V3(rDiff*scale, 0.0f, 0.0f), V3(1.0f, 0.0f, 0.0f)) 
            //the higher the scale the closer the result is to 1.0f;
            cinm__v3 color = cinm__normalized(rDiff*scale, gDiff*scale, 1.0f);

            //NOTE: convert from fp -1.0f/1.0f space to 0.0f/1.0f space then to 0u/255u space
            uint32_t r = (uint32_t)((1.0f+color.x)*0.5f*255);
            uint32_t g = (uint32_t)((1.0f+color.y)*0.5f*255);
            uint32_t b = (uint32_t)((1.0f+color.z)*0.5f*255);

            outBuffer[index] = (r | g << 8u | b << 16u | 255 << 24u);
        }
    }
}
#endif //ifndef SI_NORMALMAP_IMPLEMENTATION