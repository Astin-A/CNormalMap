/* LICENSE AT END OF FILE */

/***************************************************************************
 * Sir Irk's normal map generator
 *
 * Basic use:
 *     #define C_NORMALMAP_IMPLEMENTATION before including this file to get
        the implementation. Otherwise this acts as a regualr header file
        
 *     uint32_t *in = ...load pixels from image
 *     uint32_t *nm = cinm_normal_map(in, w, h, scale, blurRadius, greyscaleType);
 *     ...write normal map to a file
 *
 *  Other defines you can use(before including this file):
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

#ifndef _MSC_VER
#ifdef __cplusplus
#define cinm__inline inline
#else
#define cinm__inline
#endif
#else
#define cinm__inline __forceinline
#endif

#ifdef _MSC_VER
#define cinm__aligned_var(type, bytes) __declspec(align(bytes)) type
#else
#define cinm__aligned_var(type, bytes) type __attribute__((aligned(bytes)))
#endif

#ifndef CINM_TYPES
#define CINM_TYPES
typedef enum {
    cinm_greyscale_none,
    cinm_greyscale_lightness,
    cinm_greyscale_average,
    cinm_greyscale_luminance,
    cinm_greyscale_count,
} cinm_greyscale_type;
#endif //CINM_TYPES

#ifndef C_NORMALMAP_IMPLEMENTATION

CINM_DEF void cinm_greyscale(const uint32_t* in, uint32_t* out, int32_t w, int32_t h, cinm_greyscale_type type);
//Converts values in "buffer" to greyscale  using either the
//lightness, average or luminance methods
//Result can be produced in-place if "in" and "out" are the same buffers

CINM_DEF uint32_t* cinm_normal_map(const uint32_t* in, int32_t w, int32_t h, float scale, float blurRadius, cinm_greyscale_type greyscaleType, int flipY);
//Converts input buffer to a normal map and returns a pointer to it.
//  "scale" controls the intensity of the result
//  "blurRadius" controls the radius for gaussian blurring before generating normals
//  "greyscaleType" specifies the conversion method from color to greyscale before
//   generating the normal map. This step is skipped when using cinm_greyscale_none.

#else //C_NORMALMAP_IMPLEMENTATION

#include <emmintrin.h>
#include <intrin.h>

#ifdef __AVX__
#define cimd_prefix_float(name) _mm256_##name
#define CINM_CIMD_WIDTH 8
#define cimd__int __m256i
#define cimd__float __m256
#define cimd__and_ix(a, b) cimd_prefix_float(and_si256(a, b))
#define cimd__or_ix(a, b) cimd_prefix_float(or_si128(a, b))
#define cimd__loadu_ix(a) cimd_prefix_float(loadu_si256(a))
#define cimd__storeu_ix(ptr, v) cimd_prefix_float(storeu_si256(ptr, v))
#else
#define cimd_prefix_float(name) _mm_##name
#define CINM_CIMD_WIDTH 4
#define cimd__int __m128i
#define cimd__float __m128
#define cimd__and_ix(a, b) cimd_prefix_float(and_si128(a, b))
#define cimd__or_ix(a, b) cimd_prefix_float(or_si128(a, b))
#define cimd__loadu_ix(a) cimd_prefix_float(loadu_si128(a))
#define cimd__storeu_ix(ptr, v) cimd_prefix_float(storeu_si128(ptr, v))
#endif // __AVX__

#define cimd__set1_epi32(a) cimd_prefix_float(set1_epi32(a))
#define cimd__setzero_ix() cimd_prefix_float(setzero_si256())
#define cimd__setzero_ps() cimd_prefix_float(setzero_ps())
#define cimd__andnot_ps(a, b) cimd_prefix_float(andnot_ps(a, b))
#define cimd__add_epi32(a, b) cimd_prefix_float(add_epi32(a, b))
#define cimd__sub_epi32(a, b) cimd_prefix_float(sub_epi32(a, b))
#define cimd__max_epi32(a, b) cimd_prefix_float(max_epi32(a, b))
#define cimd__min_epi32(a, b) cimd_prefix_float(min_epi32(a, b))
#define cimd__loadu_ps(a) cimd_prefix_float(loadu_ps(a))
#define cimd__srli_epi32(a, i) cimd_prefix_float(srli_epi32(a, i))
#define cimd__slli_epi32(a, i) cimd_prefix_float(slli_epi32(a, i))
#define cimd__set1_ps(a) cimd_prefix_float(set1_ps(a))
#define cimd__cvtepi32_ps(a) cimd_prefix_float(cvtepi32_ps(a))
#define cimd__cvtps_epi32(a) cimd_prefix_float(cvtps_epi32(a))
#define cimd__add_ps(a, b) cimd_prefix_float(add_ps(a, b))
#define cimd__mul_ps(a, b) cimd_prefix_float(mul_ps(a, b))
#define cimd__sqrt_ps(a) cimd_prefix_float(sqrt_ps(a))
#define cimd__cmp_ps(a, b, c) cimd_prefix_float(cmp_ps(a, b, c))
#define cimd__div_ps(a, b) cimd_prefix_float(div_ps(a, b))
#define cimd__hadd_ps(a, b) cimd_prefix_float(hadd_ps(a, b))
#define cimd__cvtss_f32(a) cimd_prefix_float(cvtss_f32(a))

#define cinm__min(a, b) ((a) < (b) ? (a) : (b))
#define cinm__max(a, b) ((a) > (b) ? (a) : (b))

typedef struct
{
    int32_t x, y;
} cinm__v2i;

typedef struct
{
    float x, y, z;
} cinm__v3;

cinm__inline static float
cinm__length(float x, float y, float z)
{
    return sqrtf(x * x + y * y + z * z);
}

cinm__inline static cimd__float
cinm__length_cimd(cimd__float x, cimd__float y, cimd__float z)
{
    return cimd__sqrt_ps(cimd__add_ps(cimd__add_ps(cimd__mul_ps(x, x), cimd__mul_ps(y, y)), cimd__mul_ps(z, z)));
}

cinm__inline static cinm__v3
cinm__normalized(float x, float y, float z)
{
    cinm__v3 result;
    float len = cinm__length(x, y, z);

    if (len > 1e-04f) {
        float invLen = 1.0f / len;
        result.x = x * invLen;
        result.y = y * invLen;
        result.z = z * invLen;
    } else {
        result.x = result.y = result.z = 0.0f;
    }

    return result;
}

cinm__inline static uint32_t
cinm__lightness_average(uint32_t r, uint32_t g, uint32_t b)
{
    return (cinm__max(cinm__max(r, g), b) + cinm__min(cinm__min(r, g), b)) / 2;
}

cinm__inline static uint32_t
cinm__average(uint32_t r, uint32_t g, uint32_t b)
{
    return (r + g + b) / 3;
}

//NOTE: bias is based on human eye sensitivity
cinm__inline static uint32_t
cinm__luminance(uint32_t r, uint32_t g, uint32_t b)
{
    return (uint32_t)(0.21f * r + 0.72f * g + 0.07f * b);
}

cinm__inline static uint32_t
cinm__greyscale_from_byte(uint8_t c)
{
    return (c | c << 8u | c << 16u | 255u << 24u);
}

static cinm__inline cinm__v3
cinm__rgba_to_v3(uint32_t c)
{
    cinm__v3 result = {
        (float)((c >> 0) & 0xFFu) - 127.0f,
        (float)((c >> 8) & 0xFFu) - 127.0f,
        (float)((c >> 16) & 0xFFu) - 127.0f
    };

    return result;
}

static cinm__inline void
cinm__rgba_to_v3_cimd(cimd__int c, cimd__float* x, cimd__float* y, cimd__float* z)
{
    cimd__int ff = cimd__set1_epi32(0xFF);
    cimd__int v127 = cimd__set1_epi32(127);
    *x = cimd__cvtepi32_ps(cimd__sub_epi32(cimd__and_ix(cimd__srli_epi32(c, 0), ff), v127));
    *y = cimd__cvtepi32_ps(cimd__sub_epi32(cimd__and_ix(cimd__srli_epi32(c, 8), ff), v127));
    *z = cimd__cvtepi32_ps(cimd__sub_epi32(cimd__and_ix(cimd__srli_epi32(c, 16), ff), v127));
}

static cinm__inline uint32_t
cinm__unit_vector_to_rgba(cinm__v3 v)
{
    uint32_t r = (uint32_t)((1.0f + v.x) * 127.0f);
    uint32_t g = (uint32_t)((1.0f + v.y) * 127.0f);
    uint32_t b = (uint32_t)((1.0f + v.z) * 127.0f);
    return r | g << 8u | b << 16u | 255u << 24u;
}

static cinm__inline cimd__int
cinm__v3_to_rgba_cimd(cimd__float x, cimd__float y, cimd__float z)
{
    cimd__float one = cimd__set1_ps(1.0f);
    cimd__float v127 = cimd__set1_ps(127.0f);
    cimd__int a = cimd__set1_epi32(255u << 24u);
    cimd__int r = cimd__cvtps_epi32(cimd__mul_ps(cimd__add_ps(one, x), v127));
    cimd__int g = cimd__cvtps_epi32(cimd__mul_ps(cimd__add_ps(one, y), v127));
    cimd__int b = cimd__cvtps_epi32(cimd__mul_ps(cimd__add_ps(one, z), v127));
    cimd__int c = cimd__or_ix(cimd__or_ix(cimd__or_ix(r, cimd__slli_epi32(g, 8)), cimd__slli_epi32(b, 16)), a);
    return c;
}

CINM_DEF void
cinm__generate_gaussian_box(float* outBoxes, int32_t n, float sigma)
{
    float wIdeal = sqrtf((12.0f * sigma * sigma / (float)n) + 1.0f);
    int32_t wl = (int32_t)floorf(wIdeal);
    if (wl % 2 == 0)
        --wl;
    int32_t wu = wl + 2;

    float mIdeal = (12.0f * sigma * sigma - n * wl * wl - 4.0f * n * wl - 3.0f * n) / (-4.0f * wl - 4.0f);
    int32_t m = (int32_t)roundf(mIdeal);

    for (int i = 0; i < n; ++i) {
        outBoxes[i] = (i < m) ? (float)wl : (float)wu;
    }
}

//NOTE: decently optimized box blur based on http://blog.ivank.net/fastest-gaussian-blur.html
CINM_DEF void
cinm__box_blur_h(uint32_t* in, uint32_t* out, int32_t w, int32_t h, float r)
{
    float invR = 1.0f / (r + r + 1);
    for (int i = 0; i < h; ++i) {
        int32_t oi = i * w;
        int32_t li = oi;
        int32_t ri = (int32_t)(oi + r);
        uint32_t fv = in[oi] & 0xFFu;
        uint32_t lv = in[oi + w - 1] & 0xFFu;
        uint32_t sum = (uint32_t)((r + 1.0f) * fv);

        for (int j = 0; j < r; ++j) {
            sum += in[oi + j] & 0xFFu;
        }
        for (int j = 0; j <= r; ++j) {
            sum += (in[ri++] & 0xFFu) - fv;
            out[oi++] = cinm__greyscale_from_byte((uint8_t)(sum * invR));
        }
        for (int j = (int)r + 1; j < (w - r); ++j) {
            sum += (in[ri++] & 0xFFu) - (in[li++] & 0xFFu);
            out[oi++] = cinm__greyscale_from_byte((uint8_t)(sum * invR));
        }
        for (int j = (int)(w - r); j < w; ++j) {
            sum += lv - (in[li++] & 0xFFu);
            out[oi++] = cinm__greyscale_from_byte((uint8_t)(sum * invR));
        }
    }
}

CINM_DEF void
cinm__box_blur_v(uint32_t* in, uint32_t* out, int32_t w, int32_t h, float r)
{
    float invR = 1.0f / (r + r + 1);
    for (int i = 0; i < w; ++i) {
        int32_t oi = i;
        int32_t li = oi;
        int32_t ri = (int32_t)(oi + r * w);
        uint32_t fv = in[oi] & 0xFFu;
        uint32_t lv = in[oi + w * (h - 1)] & 0xFFu;
        uint32_t sum = (uint32_t)((r + 1) * fv);

        for (int j = 0; j < r; j++) {
            sum += in[oi + j * w] & 0xFFu;
        }
        for (int j = 0; j <= r; j++) {
            sum += (in[ri] & 0xFFu) - fv;
            out[oi] = cinm__greyscale_from_byte((uint8_t)(sum * invR));
            ri += w;
            oi += w;
        }
        for (int j = (int)(r + 1); j < h - r; j++) {
            sum += (in[ri] & 0xFFu) - (in[li] & 0xFFu);
            out[oi] = cinm__greyscale_from_byte((uint8_t)(sum * invR));
            li += w;
            ri += w;
            oi += w;
        }
        for (int j = (int)(h - r); j < h; j++) {
            sum += lv - (in[li] & 0xFFu);
            out[oi] = cinm__greyscale_from_byte((uint8_t)(sum * invR));
            li += w;
            oi += w;
        }
    }
}

CINM_DEF void
cinm__gaussian_box(uint32_t* in, uint32_t* out, int32_t w, int32_t h, float r)
{
    float boxes[3];
    cinm__generate_gaussian_box(boxes, sizeof(boxes) / sizeof(boxes[0]), r);

    for (int i = 0; i < 3; ++i) {
        cinm__box_blur_h(in, out, w, h, (boxes[i] - 1) / 2);
        cinm__box_blur_v(out, in, w, h, (boxes[i] - 1) / 2);
    }

    memcpy(out, in, w * h * sizeof(uint32_t));
}

CINM_DEF void
cinm__sobel3x3_normals_row_range(const uint32_t* in, uint32_t* out, int32_t xs, int32_t xe, int32_t w, int32_t h, float scale, int flipY)
{
    const float xk[3][3] = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 },
    };
    const float yk[3][3] = {
        { -1, -2, -1 },
        { 0, 0, 0 },
        { 1, 2, 1 },
    };

    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = xs; x < xe; ++x) {
            float xmag = 0.0f;
            float ymag = 0.0f;
            for (int32_t a = 0; a < 3; ++a) {
                for (int32_t b = 0; b < 3; ++b) {
                    int32_t xIdx = cinm__min(w - 1, cinm__max(1, x + b - 1));
                    int32_t yIdx = cinm__min(h - 1, cinm__max(1, y + a - 1));
                    int32_t index = yIdx * w + xIdx;
                    uint32_t pixel = in[index] & 0xFFu;
                    xmag += pixel * xk[a][b];
                    ymag += pixel * yk[a][b];
                }
            }
            cinm__v3 color = cinm__normalized(xmag * scale, ymag * scale, 255.0f);
            out[y * w + x] = cinm__unit_vector_to_rgba(color);
        }
    }
}

static cinm__inline void
cinm__sobel3x3_normals(const uint32_t* in, uint32_t* out, int32_t w, int32_t h, float scale, int flipY)
{
    cinm__sobel3x3_normals_row_range(in, out, 0, w, w, h, scale, flipY);
}

static void
cinm__sobel3x3_normals_cimd(const uint32_t* in, uint32_t* out, int32_t w, int32_t h, float scale, int flipY)
{
    const float xk[3][4] = {
        { -1, 0, 1, 0 },
        { -2, 0, 2, 0 },
        { -1, 0, 1, 0 },
    };
    const float yk[3][4] = {
        { -1, -2, -1, 0 },
        { 0, 0, 0, 0 },
        { 1, 2, 1, 0 },
    };

    cimd__float cimdScale = cimd__set1_ps(scale);
    cimd__float cimdFlipY = cimd__set1_ps((flipY) ? -1.0f : 1.0f);
    cimd__float cimd1 = cimd__set1_ps(1.0f);
    cimd__float cimd127 = cimd__set1_ps(127.0f);

    int32_t batchCounter = 0;
    cinm__aligned_var(float, CINM_CIMD_WIDTH) xBatch[CINM_CIMD_WIDTH];
    cinm__aligned_var(float, CINM_CIMD_WIDTH) yBatch[CINM_CIMD_WIDTH];

    for (int32_t yIter = 0; yIter < h; ++yIter) {
        for (int32_t xIter = CINM_CIMD_WIDTH; xIter < w - CINM_CIMD_WIDTH; ++xIter) {
            __m128 xmag = _mm_set1_ps(0.0f);
            __m128 ymag = _mm_set1_ps(0.0f);

            for (int32_t a = 0; a < 3; ++a) {
                int32_t xIdx = cinm__min(w - 1, cinm__max(1, xIter - 1));
                int32_t yIdx = cinm__min(h - 1, cinm__max(1, yIter + a - 1));
                int32_t index = yIdx * w + xIdx;

                __m128i pixel = _mm_loadu_si128((__m128i*)&in[index]);
                pixel = _mm_and_si128(pixel, _mm_set1_epi32(0xFFu));
                __m128 pixelf = _mm_cvtepi32_ps(pixel);
                __m128 kx = _mm_loadu_ps((float*)&xk[a]);
                __m128 ky = _mm_loadu_ps((float*)&yk[a]);
                xmag = _mm_add_ps(_mm_mul_ps(pixelf, kx), xmag);
                ymag = _mm_add_ps(_mm_mul_ps(pixelf, ky), ymag);
            }

            __m128 xSum = _mm_hadd_ps(xmag, xmag);
            __m128 ySum = _mm_hadd_ps(ymag, ymag);
            float xn = _mm_cvtss_f32(_mm_hadd_ps(xSum, xSum));
            float yn = _mm_cvtss_f32(_mm_hadd_ps(ySum, ySum));

            xBatch[batchCounter] = xn;
            yBatch[batchCounter++] = yn;
            if (batchCounter == CINM_CIMD_WIDTH) {
                batchCounter = 0;
                cimd__float x = cimd__loadu_ps(xBatch);
                cimd__float y = cimd__loadu_ps(yBatch);
                cimd__float z = cimd__set1_ps(255.0f);

                x = cimd__mul_ps(x, cimdScale);
                y = cimd__mul_ps(y, cimdScale);

                //normalize
                cimd__float len = cinm__length_cimd(x, y, z);
                cimd__float invLen = cimd__div_ps(cimd__set1_ps(1.0f), len);
                x = cimd__mul_ps(x, invLen);
                y = cimd__mul_ps(y, invLen);
                z = cimd__mul_ps(z, invLen);

                int index = yIter * w + (xIter - (CINM_CIMD_WIDTH - 1));
                cimd__storeu_ix((cimd__int*)&out[index], cinm__v3_to_rgba_cimd(x, y, z));
            }
        }
    }

    cinm__sobel3x3_normals_row_range(in, out, 0, CINM_CIMD_WIDTH, w, h, scale, flipY);
    cinm__sobel3x3_normals_row_range(in, out, w - CINM_CIMD_WIDTH, w, w, h, scale, flipY);
}

CINM_DEF void
cinm_normalize(uint32_t* in, int32_t w, int32_t h, float scale, int flipY)
{
    float invScale = 1.0f / scale;
    float yDir = (flipY) ? -1.0f : 1.0f;
    for (int32_t i = 0; i < w * h; ++i) {
        cinm__v3 v = cinm__rgba_to_v3(in[i]);
        in[i] = cinm__unit_vector_to_rgba(cinm__normalized(v.x, v.y * yDir, v.z * invScale));
    }
}

CINM_DEF void
cinm__normalize_cimd(uint32_t* in, int32_t w, int32_t h, float scale, int flipY)
{
    assert(w % CINM_CIMD_WIDTH == 0);
    for (int32_t i = 0; i < w * h; i += CINM_CIMD_WIDTH) {
        cimd__int pixel = cimd__loadu_ix((cimd__int*)&in[i]);
        cimd__float x, y, z;
        cinm__rgba_to_v3_cimd(pixel, &x, &y, &z);
        cimd__float len = cinm__length_cimd(x, y, z);
        cimd__float invLen = cimd__div_ps(cimd__set1_ps(1.0f), len);
        cimd__float mask = cimd__cmp_ps(cimd__setzero_ps(), len, _CMP_EQ_OQ);
        x = cimd__andnot_ps(mask, cimd__mul_ps(x, invLen));
        y = cimd__andnot_ps(mask, cimd__mul_ps(y, invLen));
        z = cimd__andnot_ps(mask, cimd__mul_ps(z, invLen));
        cimd__storeu_ix((cimd__int*)&in[i], cinm__v3_to_rgba_cimd(x, y, z));
    }
}

CINM_DEF void
cinm_composite(const uint32_t* in, uint32_t* out, int32_t w, int32_t h)
{
    for (int32_t i = 0; i < w * h; ++i) {
        uint32_t ic = in[i], oc = out[i];
        uint32_t r = (uint32_t)((((ic >> 0) & 0xFFu) + ((oc >> 0) & 0xFFu)) * 0.5f);
        uint32_t g = (uint32_t)((((ic >> 8) & 0xFFu) + ((oc >> 8) & 0xFFu)) * 0.5f);
        uint32_t b = (uint32_t)((((ic >> 16) & 0xFFu) + ((oc >> 16) & 0xFFu)) * 0.5f);
        out[i] = (r | g << 8u | b << 16u | 255u << 24u);
    }
}

static void
cinm__greyscale(const uint32_t* in, uint32_t* out, int32_t w, int32_t h, cinm_greyscale_type type)
{
    int32_t count = w * h;
    switch (type) {
    case cinm_greyscale_lightness: {
        for (int32_t i = 0; i < count; ++i) {
            uint32_t c = in[i];
            uint32_t l = cinm__lightness_average(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu);
            out[i] = cinm__greyscale_from_byte(l);
        }
    } break;

    case cinm_greyscale_average: {
        for (int32_t i = 0; i < count; ++i) {
            uint32_t c = in[i];
            uint32_t l = cinm__average(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu);
            out[i] = cinm__greyscale_from_byte(l);
        }
    } break;

    case cinm_greyscale_luminance: {
        for (int32_t i = 0; i < count; ++i) {
            uint32_t c = in[i];
            uint32_t l = cinm__luminance(c & 0xFFu, (c >> 8) & 0xFFu, (c >> 16) & 0xFFu);
            out[i] = cinm__greyscale_from_byte(l);
        }
    } break;
    }
}

static void
cinm__cimd_greyscale(const uint32_t* in, uint32_t* out, int32_t w, int32_t h, cinm_greyscale_type type)
{
    cimd__int redMask = cimd__set1_epi32(0xFF);
    cimd__int greenMask = cimd__set1_epi32(0xFF00u);
    cimd__int blueMask = cimd__set1_epi32(0xFF0000u);
    cimd__int alpha = cimd__set1_epi32(0xFF000000u);

    int32_t count = w * h;

    switch (type) {
    case cinm_greyscale_lightness: {
        for (int32_t i = 0; i < count; i += CINM_CIMD_WIDTH) {
            cimd__int c = cimd__loadu_ix((cimd__int*)&in[i]);
            cimd__int r = cimd__and_ix(c, redMask);
            cimd__int g = cimd__srli_epi32(cimd__and_ix(c, greenMask), 8);
            cimd__int b = cimd__srli_epi32(cimd__and_ix(c, blueMask), 16);

            cimd__int max = cimd__max_epi32(cimd__max_epi32(r, g), b);
            cimd__int min = cimd__min_epi32(cimd__min_epi32(r, g), b);
            cimd__int l = cimd__srli_epi32(cimd__add_epi32(min, max), 1);

            l = cimd__or_ix(cimd__slli_epi32(l, 16),
                cimd__or_ix(cimd__slli_epi32(l, 8),
                    cimd__or_ix(l, alpha)));

            cimd__storeu_ix((cimd__int*)&out[i], l);
        }
    } break;

    case cinm_greyscale_average: {
        cimd__float inverse3 = cimd__set1_ps(1.0f / 3.0f);
        for (int32_t i = 0; i < count; i += CINM_CIMD_WIDTH) {
            cimd__int c = cimd__loadu_ix((cimd__int*)&in[i]);
            cimd__int r = cimd__and_ix(c, redMask);
            cimd__int g = cimd__srli_epi32(cimd__and_ix(c, greenMask), 8);
            cimd__int b = cimd__srli_epi32(cimd__and_ix(c, blueMask), 16);

            cimd__int s = cimd__add_epi32(cimd__add_epi32(r, g), b);
            s = cimd__cvtps_epi32(cimd__mul_ps(cimd__cvtepi32_ps(s), inverse3));
            s = cimd__or_ix(cimd__slli_epi32(s, 16),
                cimd__or_ix(cimd__slli_epi32(s, 8),
                    cimd__or_ix(s, alpha)));

            cimd__storeu_ix((cimd__int*)&out[i], s);
        }
    } break;

    case cinm_greyscale_luminance: {
        cimd__float rBias = cimd__set1_ps(0.21f);
        cimd__float gBias = cimd__set1_ps(0.72f);
        cimd__float bBias = cimd__set1_ps(0.07f);

        for (int32_t i = 0; i < count; i += CINM_CIMD_WIDTH) {
            cimd__int c = cimd__loadu_ix((cimd__int*)&in[i]);
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

            cimd__storeu_ix((cimd__int*)&out[i], sum);
        }
    } break;
    }
}

CINM_DEF void
cinm_greyscale(const uint32_t* in, uint32_t* out, int32_t w, int32_t h, cinm_greyscale_type type)
{
    int32_t count = w * h;
    if (count % CINM_CIMD_WIDTH == 0) {
        cinm__cimd_greyscale(in, out, w, h, type);
    } else {
        cinm__greyscale(in, out, w, h, type);
    }
}

CINM_DEF int
cinm_normal_map_buffer(const uint32_t* in, uint32_t* out, int32_t w, int32_t h, float scale, float blurRadius, cinm_greyscale_type greyscaleType, int flipY, int useSimd)
{
    assert(w > 0 && h > 0);
    uint32_t* intermediate = (uint32_t*)malloc(w * h * sizeof(uint32_t));

    if (intermediate) {
        if (greyscaleType != cinm_greyscale_none) {
            cinm_greyscale(in, out, w, h, greyscaleType);
        } else {
            memcpy(out, in, w * h * sizeof(uint32_t));
        }

        float radius = cinm__min(cinm__min(w, h), cinm__max(0, blurRadius));
        if (radius >= 1.0f) {
            cinm__gaussian_box(out, intermediate, w, h, radius);
        } else {
            memcpy(intermediate, out, w * h * sizeof(uint32_t));
        }

        //TODO: support using cimd on non power of 2 images
        int32_t count = w * h;
        if (count % CINM_CIMD_WIDTH == 0) {
            cinm__sobel3x3_normals_cimd(intermediate, out, w, h, scale, flipY);
        } else {
            cinm__sobel3x3_normals(intermediate, out, w, h, scale, flipY);
        }

        free(intermediate);
        return 1;
    }
    return 0;
}

CINM_DEF uint32_t*
cinm_normal_map(const uint32_t* in, int32_t w, int32_t h, float scale, float blurRadius, cinm_greyscale_type greyscaleType, int flipY, int useSimd)
{
    uint32_t* result = (uint32_t*)malloc(w * h * sizeof(uint32_t));
    if (result) {
        if (!cinm_normal_map_buffer(in, result, w, h, scale, blurRadius, greyscaleType, flipY, useSimd)) {
            free(result);
            return NULL;
        }
    }
    return result;
}

#endif //ifndef C_NORMALMAP_IMPLEMENTATION
