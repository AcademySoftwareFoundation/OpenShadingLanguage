// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// clang-format off

#pragma once

#include <limits>

#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/sfmath.h>
#include <OSL/wide.h>
#include <OpenImageIO/hash.h>
#include <OpenImageIO/simd.h>

OSL_NAMESPACE_ENTER


///////////////////////////////////////////////////////////////////////
// Implementation follows...
//
// Users don't need to worry about this part
///////////////////////////////////////////////////////////////////////

#ifdef __OSL_WIDE_PVT
    namespace __OSL_WIDE_PVT {
#else
    namespace pvt {
#endif

//} // pvt


// non SIMD version, should be scalar code meant to be used
// inside SIMD loops
// SIMD FRIENDLY MATH
namespace sfm
{

// Code to convert lookup tables to masks for 1.0f and -1.0f (default 0.0f) and spit
// the resulting masks out to std::cout to be cut n pasted back as constants.
// to be able to
//#define __OSL_EMIT_GRAD_MASKS 1
#ifdef __OSL_EMIT_GRAD_MASKS

template<int N>
static void emit_grad_mask(const char * name, const float (&vals)[N], float target) {

    static_assert(N <= 64, "N too large");
    typedef typename std::conditional<N <= 16,
                                      uint16_t,
                                      uint32_t>::type Mask16or32;
    typedef typename std::conditional<N <= 8,
                                      uint8_t,
                                      Mask16or32>::type MaskType;
    const char * name_of_mask_type = "uint32_t";
    // Assume RTTI is disabled, so compile time tests to choose the correct type name
    if (std::is_same<MaskType, uint16_t>::value)
        name_of_mask_type = "uint16_t";
    if (std::is_same<MaskType, uint8_t>::value)
        name_of_mask_type = "uint8_t";
    std::cout << "//static constexpr " << name_of_mask_type << " " << name << " = 0b";
    MaskType mask = static_cast<MaskType>(0u);
    for(int index=N-1; index >=0; --index) {
        float val = vals[index];
        const char * result = "0";
        if (val == target) {
            result = "1";
            mask |= MaskType(1u)<<index;
        }
        std::cout << result;
    }
    std::cout << ";" << std::endl;
    // NOTE: use of unary + operator to upconvert uint8_t to not be output as a char
    std::cout << "static constexpr " << name_of_mask_type << " " << name << " = 0x"
              << std::hex << std::setw(2*sizeof(MaskType)) << std::uppercase  << std::setfill('0') << +mask << std::nouppercase << std::dec << ";" << std::endl;
}
#endif

// Gradient table for 2D. These could be programmed the Ken Perlin way with
// some clever bit-twiddling, but this is more clear, and not really slower.
namespace fast_grad2 {
    alignas(64) static const OSL::Block<Vec2,8> lut_wide(
        Vec2( -1.0f, -1.0f ), Vec2( 1.0f,  0.0f ), Vec2( -1.0f, 0.0f ), Vec2( 1.0f,  1.0f ),
        Vec2( -1.0f,  1.0f ), Vec2( 0.0f, -1.0f ), Vec2(  0.0f, 1.0f ), Vec2(  1.0f, -1.0f ));

    //static constexpr uint8_t x_one_mask     = 0b10001010;
    static constexpr uint8_t x_one_mask       = 0x8A;
    //static constexpr uint8_t x_neg_one_mask = 0b00010101;
    static constexpr uint8_t x_neg_one_mask   = 0x15;
    //static constexpr uint8_t y_one_mask     = 0b01011000;
    static constexpr uint8_t y_one_mask       = 0x58;
    //static constexpr uint8_t y_neg_one_mask = 0b10100001;
    static constexpr uint8_t y_neg_one_mask   = 0xA1;

#ifdef __OSL_EMIT_GRAD_MASKS
    class OSLEXECPUBLIC mask_emitter {
    public:
        mask_emitter() {
            static bool has_emitted = false;
            if (!has_emitted) {
                has_emitted = true;
                emit_masks();
            }
        }
    private:
        void emit_masks()
        {
            emit_grad_mask("x_one_mask", lut_wide.x, 1.0f);
            emit_grad_mask("x_neg_one_mask", lut_wide.x, -1.0f);

            emit_grad_mask("y_one_mask", lut_wide.y, 1.0f);
            emit_grad_mask("y_neg_one_mask", lut_wide.y, -1.0f);
        }
    };
#endif
}


// Gradient directions for 3D.
// These vectors are based on the midpoints of the 12 edges of a cube.
// A larger array of random unit length vectors would also do the job,
// but these 12 (including 4 repeats to make the array length a power
// of two) work better. They are not random, they are carefully chosen
// to represent a small, isotropic set of directions.

namespace fast_grad3 {
    // Store in SOA data layout using our Block helper
    alignas(64) static const OSL::Block<Vec3,16> lut_wide(
        Vec3(  1.0f,  0.0f,  1.0f ), Vec3(  0.0f,  1.0f,  1.0f ), // 12 cube edges
        Vec3( -1.0f,  0.0f,  1.0f ), Vec3( 0.0f, -1.0f,  1.0f ),
        Vec3(  1.0f,  0.0f, -1.0f ), Vec3( 0.0f,  1.0f, -1.0f ),
        Vec3( -1.0f,  0.0f, -1.0f ), Vec3(  0.0f, -1.0f, -1.0f ),
        Vec3(  1.0f, -1.0f,  0.0f ), Vec3( 1.0f,  1.0f,  0.0f ),
        Vec3( -1.0f,  1.0f,  0.0f ), Vec3( -1.0f, -1.0f,  0.0f ),
        Vec3(  1.0f,  0.0f,  1.0f ), Vec3( -1.0f,  0.0f,  1.0f ), // 4 repeats to make 16
        Vec3(  0.0f,  1.0f, -1.0f ), Vec3( 0.0f, -1.0f, -1.0f ));

    //static constexpr uint16_t x_one_mask = 0b0001001100010001;
    static constexpr uint16_t x_one_mask = 0x1311;
    //static constexpr uint16_t x_neg_one_mask = 0b0010110001000100;
    static constexpr uint16_t x_neg_one_mask = 0x2C44;

    //static constexpr uint16_t y_one_mask = 0b0100011000100010;
    static constexpr uint16_t y_one_mask = 0x4622;
    //static constexpr uint16_t y_neg_one_mask = 0b1000100110001000;
    static constexpr uint16_t y_neg_one_mask = 0x8988;

    //static constexpr uint16_t z_one_mask = 0b0011000000001111;
    static constexpr uint16_t z_one_mask = 0x300F;
    //static constexpr uint16_t z_neg_one_mask = 0b1100000011110000;
    static constexpr uint16_t z_neg_one_mask = 0xC0F0;

#ifdef __OSL_EMIT_GRAD_MASKS
    class OSLEXECPUBLIC mask_emitter {
    public:
        mask_emitter() {
            static bool has_emitted = false;
            if (!has_emitted) {
                has_emitted = true;
                emit_masks();
            }
        }
    private:
        void emit_masks()
        {
            emit_grad_mask("x_one_mask", lut_wide.x, 1.0f);
            emit_grad_mask("x_neg_one_mask", lut_wide.x, -1.0f);

            emit_grad_mask("y_one_mask", lut_wide.y, 1.0f);
            emit_grad_mask("y_neg_one_mask", lut_wide.y, -1.0f);

            emit_grad_mask("z_one_mask", lut_wide.z, 1.0f);
            emit_grad_mask("z_neg_one_mask", lut_wide.z, -1.0f);
        }
    };
#endif
}

//static fast_grad3::mask_validator the_fast_grad3_mask_validator;

namespace fast_grad4 {
    alignas(64) static const OSL::Block<Vec4,32> lut_wide(
      Vec4( 0.0f, 1.0f, 1.0f, 1.0f ),  Vec4( 0.0f, 1.0f, 1.0f, -1.0f ),  Vec4( 0.0f, 1.0f, -1.0f, 1.0f ),  Vec4( 0.0f, 1.0f, -1.0f, -1.0f ), // 32 tesseract edges
      Vec4( 0.0f, -1.0f, 1.0f, 1.0f ), Vec4( 0.0f, -1.0f, 1.0f, -1.0f ), Vec4( 0.0f, -1.0f, -1.0f, 1.0f ), Vec4( 0.0f, -1.0f, -1.0f, -1.0f ),
      Vec4( 1.0f, 0.0f, 1.0f, 1.0f ),  Vec4( 1.0f, 0.0f, 1.0f, -1.0f ),  Vec4( 1.0f, 0.0f, -1.0f, 1.0f ),  Vec4( 1.0f, 0.0f, -1.0f, -1.0f ),
      Vec4( -1.0f, 0.0f, 1.0f, 1.0f ), Vec4( -1.0f, 0.0f, 1.0f, -1.0f ), Vec4( -1.0f, 0.0f, -1.0f, 1.0f ), Vec4( -1.0f, 0.0f, -1.0f, -1.0f ),
      Vec4( 1.0f, 1.0f, 0.0f, 1.0f ),  Vec4( 1.0f, 1.0f, 0.0f, -1.0f ),  Vec4( 1.0f, -1.0f, 0.0f, 1.0f ),  Vec4( 1.0f, -1.0f, 0.0f, -1.0f ),
      Vec4( -1.0f, 1.0f, 0.0f, 1.0f ), Vec4( -1.0f, 1.0f, 0.0f, -1.0f ), Vec4( -1.0f, -1.0f, 0.0f, 1.0f ), Vec4( -1.0f, -1.0f, 0.0f, -1.0f ),
      Vec4( 1.0f, 1.0f, 1.0f, 0.0f ),  Vec4( 1.0f, 1.0f, -1.0f, 0.0f ),  Vec4( 1.0f, -1.0f, 1.0f, 0.0f ),  Vec4( 1.0f, -1.0f, -1.0f, 0.0f ),
      Vec4( -1.0f, 1.0f, 1.0f, 0.0f ), Vec4( -1.0f, 1.0f, -1.0f, 0.0f ), Vec4( -1.0f, -1.0f, 1.0f, 0.0f ), Vec4( -1.0f, -1.0f, -1.0f, 0.0f ));


    //static constexpr uint32_t x_one_mask     = 0b00001111000011110000111100000000;
    static constexpr uint32_t x_one_mask       = 0x0F0F0F00;
    //static constexpr uint32_t x_neg_one_mask = 0b11110000111100001111000000000000;
    static constexpr uint32_t x_neg_one_mask   = 0xF0F0F000;
    //static constexpr uint32_t y_one_mask     = 0b00110011001100110000000000001111;
    static constexpr uint32_t y_one_mask       = 0x3333000F;
    //static constexpr uint32_t y_neg_one_mask = 0b11001100110011000000000011110000;
    static constexpr uint32_t y_neg_one_mask   = 0xCCCC00F0;
    //static constexpr uint32_t z_one_mask     = 0b01010101000000000011001100110011;
    static constexpr uint32_t z_one_mask       = 0x55003333;
    //static constexpr uint32_t z_neg_one_mask = 0b10101010000000001100110011001100;
    static constexpr uint32_t z_neg_one_mask   = 0xAA00CCCC;
    //static constexpr uint32_t w_one_mask     = 0b00000000010101010101010101010101;
    static constexpr uint32_t w_one_mask       = 0x00555555;
    //static constexpr uint32_t w_neg_one_mask = 0b00000000101010101010101010101010;
    static constexpr uint32_t w_neg_one_mask   = 0x00AAAAAA;


#ifdef __OSL_EMIT_GRAD_MASKS
    class OSLEXECPUBLIC mask_emitter {
    public:
        mask_emitter() {
            static bool has_emitted = false;
            if (!has_emitted) {
                has_emitted = true;
                emit_masks();
            }
        }
    private:
        void emit_masks()
        {
            emit_grad_mask("x_one_mask", lut_wide.x, 1.0f);
            emit_grad_mask("x_neg_one_mask", lut_wide.x, -1.0f);

            emit_grad_mask("y_one_mask", lut_wide.y, 1.0f);
            emit_grad_mask("y_neg_one_mask", lut_wide.y, -1.0f);

            emit_grad_mask("z_one_mask", lut_wide.z, 1.0f);
            emit_grad_mask("z_neg_one_mask", lut_wide.z, -1.0f);

            emit_grad_mask("w_one_mask", lut_wide.w, 1.0f);
            emit_grad_mask("w_neg_one_mask", lut_wide.w, -1.0f);
        }
    };
#endif

}


namespace fast_simplex {
    static const unsigned char lut[64][4] = {
      {0,1,2,3},{0,1,3,2},{0,0,0,0},{0,2,3,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,2,3,0},
      {0,2,1,3},{0,0,0,0},{0,3,1,2},{0,3,2,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,3,2,0},
      {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
      {1,2,0,3},{0,0,0,0},{1,3,0,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,3,0,1},{2,3,1,0},
      {1,0,2,3},{1,0,3,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,0,3,1},{0,0,0,0},{2,1,3,0},
      {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
      {2,0,1,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,0,1,2},{3,0,2,1},{0,0,0,0},{3,1,2,0},
      {2,1,0,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,1,0,2},{0,0,0,0},{3,2,0,1},{3,2,1,0}};


    //static constexpr uint64_t corner2_i_offset_mask = 0b1101000010110000000000000000000000000000000000000000000000000000;
    static constexpr uint64_t corner2_i_offset_mask   = 0xD0B0000000000000;
    //static constexpr uint64_t corner2_j_offset_mask = 0b0000000000000000000000000000000011000100000000001000110000000000;
    static constexpr uint64_t corner2_j_offset_mask   = 0x00000000C4008C00;
    //static constexpr uint64_t corner2_k_offset_mask = 0b0000000000000000000000001010001000000000000000000000000010001010;
    static constexpr uint64_t corner2_k_offset_mask   = 0x000000A20000008A;
    //static constexpr uint64_t corner2_l_offset_mask = 0b0000000100000001000000000000000100000001000000000000000100000001;
    static constexpr uint64_t corner2_l_offset_mask   = 0x0101000101000101;
    //static constexpr uint64_t corner3_i_offset_mask = 0b1101000110110001000000001010000011000000000000000000000000000000;
    static constexpr uint64_t corner3_i_offset_mask   = 0xD1B100A0C0000000;
    //static constexpr uint64_t corner3_j_offset_mask = 0b1100000000000000000000000000000011000101000000001000110110001000;
    static constexpr uint64_t corner3_j_offset_mask   = 0xC0000000C5008D88;
    //static constexpr uint64_t corner3_k_offset_mask = 0b0000000010100000000000001010001100000000000000001000100010001011;
    static constexpr uint64_t corner3_k_offset_mask   = 0x00A000A30000888B;
    //static constexpr uint64_t corner3_l_offset_mask = 0b0001000100010001000000000000001100000101000000000000010100000011;
    static constexpr uint64_t corner3_l_offset_mask   = 0x1111000305000503;
    //static constexpr uint64_t corner4_i_offset_mask = 0b1101000110110001000000001010001111000101000000001000000010000000;
    static constexpr uint64_t corner4_i_offset_mask   = 0xD1B100A3C5008080;
    //static constexpr uint64_t corner4_j_offset_mask = 0b1101000110000000000000001000000011000101000000001000110110001011;
    static constexpr uint64_t corner4_j_offset_mask   = 0xD1800080C5008D8B;
    //static constexpr uint64_t corner4_k_offset_mask = 0b1000000010110001000000001010001110000000000000001000110110001011;
    static constexpr uint64_t corner4_k_offset_mask   = 0x80B100A380008D8B;
    //static constexpr uint64_t corner4_l_offset_mask = 0b0101000100110001000000000010001101000101000000000000110100001011;
    static constexpr uint64_t corner4_l_offset_mask   = 0x5131002345000D0B;


#ifdef __OSL_EMIT_GRAD_MASKS
    class OSLEXECPUBLIC mask_emitter {
    public:
        mask_emitter() {
            static bool has_emitted = false;
            if (!has_emitted) {
                has_emitted = true;
                emit_masks();
            }
        }
    private:

        static void emit_mask(const char * name, int vec_index, unsigned char threshold) {
            std::cout << "//static constexpr uint64_t " << name << " = 0b";
            uint64_t mask = 0u;
            for(int index=64-1; index >=0; --index) {
                unsigned char val = lut[index][vec_index];
                const char * result = "0";
                if (val >= threshold) {
                    result = "1";
                    mask |= uint64_t(1u)<<index;
                }
                std::cout << result;
            }
            std::cout << ";" << std::endl;
            std::cout << "static constexpr uint64_t " << name << " = 0x"
                      << std::hex << std::setw(2*sizeof(uint64_t)) << std::uppercase  << std::setfill('0') << mask << std::nouppercase << std::dec << ";" << std::endl;
        }

        void emit_masks()
        {
            emit_mask("corner2_i_offset_mask", 0, 3);
            emit_mask("corner2_j_offset_mask", 1, 3);
            emit_mask("corner2_k_offset_mask", 2, 3);
            emit_mask("corner2_l_offset_mask", 3, 3);

            emit_mask("corner3_i_offset_mask", 0, 2);
            emit_mask("corner3_j_offset_mask", 1, 2);
            emit_mask("corner3_k_offset_mask", 2, 2);
            emit_mask("corner3_l_offset_mask", 3, 2);

            emit_mask("corner4_i_offset_mask", 0, 1);
            emit_mask("corner4_j_offset_mask", 1, 1);
            emit_mask("corner4_k_offset_mask", 2, 1);
            emit_mask("corner4_l_offset_mask", 3, 1);
        }
    };
#endif
}

#ifdef __OSL_EMIT_GRAD_MASKS
#ifndef __OSL_WIDE_PVT
// Only emit grad masks for non-wide namespace as the wide would end up generating multiple sets of classes
// each with its own local static variables and emit the masks multiple times
static fast_grad2::mask_emitter the_fast_grad2_mask_emitter;
static fast_grad3::mask_emitter the_fast_grad3_mask_emitter;
static fast_grad4::mask_emitter the_fast_grad4_mask_emitter;
static fast_simplex::mask_emitter the_fast_simplex_mask_emitter;
#endif
#endif

//#define OSL_VERIFY_SIMPLEX3 1
#ifdef OSL_VERIFY_SIMPLEX3
        static void osl_verify_fail(int lineNumber, const char *expression)
        {
            std::cout << "Line " << __LINE__ << " failed OSL_VERIFY(" << expression << ")" << std::endl;
            exit(1);
        }
    #define OSL_VERIFY(EXPR) if((EXPR)== false) osl_verify_fail(__LINE__, #EXPR);
#endif


    static OSL_FORCEINLINE uint32_t
    scramble (uint32_t v0, uint32_t v1=0, uint32_t v2=0)
    {
        return OIIO::bjhash::bjfinal (v0, v1, v2^0xdeadbeef);
    }


    template<int seedT>
    static OSL_FORCEINLINE float
    grad1 (int i)
    {
        int h = scramble (i, seedT);
        float g = 1.0f + (h & 7);   // Gradient value is one of 1.0, 2.0, ..., 8.0
        if (h & 8)
            g = -g;   // Make half of the gradients negative
        return g;
    }

    OSL_FORCEINLINE float select(const bool b, float t, float f) { return b ? t : f; }
    OSL_FORCEINLINE float negate_if (const float val, const bool b) {
        return b ? -val : val;
    }

    // To use the old (slower) lookup table based grad, just build with -D__OSL_LUT_GRAD=1
#ifndef __OSL_LUT_GRAD
    #define __OSL_LUT_GRAD 0
#endif

    template<int seedT>
    static OSL_FORCEINLINE Vec2
    grad2 (int i, int j)
    {
        int h = scramble (i, j, seedT);
    #if __OSL_LUT_GRAD
        //return grad2lut[h & 7];
        return fast_grad2::lut_wide.get(h & 7);
    #else
        int clamped_index = h&7;
        uint8_t bit_index = uint8_t(1) << clamped_index;
        float x = 0.0f;
        if (bit_index & fast_grad2::x_one_mask)
            x = 1.0f;
        if (bit_index & fast_grad2::x_neg_one_mask)
            x = -1.0f;

        float y = 0.0f;
        if (bit_index & fast_grad2::y_one_mask)
            y = 1.0f;
        if (bit_index & fast_grad2::y_neg_one_mask)
            y = -1.0f;
        return Vec2(x,y);
    #endif
    }

    template<int seedT>
    static OSL_FORCEINLINE Vec3
    grad3 (int i, int j, int k)
    {
        int h = scramble (i, j, scramble (k, seedT));
#if __OSL_LUT_GRAD
        //return fast_grad3lut[h & 15];
        return fast_grad3::lut_wide.get(h & 15);
#else
        // Clang was unhappy trying to SIMD a
        // table lookup, so turn index into a linear bit index
        // and use a mask to blend 1.0f and -1.0f on the appropriate indices
        //return fast_grad3::lut_wide.get(h & 15);
        int clamped_index = h&15;
        uint16_t bit_index = uint16_t(1) << clamped_index;
        float x = 0.0f;
        if (bit_index & fast_grad3::x_one_mask)
            x = 1.0f;
        if (bit_index & fast_grad3::x_neg_one_mask)
            x = -1.0f;

        float y = 0.0f;
        if (bit_index & fast_grad3::y_one_mask)
            y = 1.0f;
        if (bit_index & fast_grad3::y_neg_one_mask)
            y = -1.0f;

        float z = 0.0f;
        if (bit_index & fast_grad3::z_one_mask)
            z = 1.0f;
        if (bit_index & fast_grad3::z_neg_one_mask)
            z = -1.0f;

        return Vec3(x,y,z);
#endif
    }

    template<int seedT>
    static OSL_FORCEINLINE const Vec4
    grad4 (int i, int j, int k, int l)
    {
        int h = scramble (i, j, scramble (k, l, seedT));
#if __OSL_LUT_GRAD
        // return fast_grad4lut[h & 31];
        return fast_grad4::lut_wide.get(h & 31);
#else
        // Clang was unhappy trying to SIMD a
        // table lookup, so turn index into a linear bit index
        // and use a mask to blend 1.0f and -1.0f on the appropriate indices
        int clamped_index = h&31;
        uint32_t bit_index = uint32_t(1) << clamped_index;
        float x = 0.0f;
        if (bit_index & fast_grad4::x_one_mask)
            x = 1.0f;
        if (bit_index & fast_grad4::x_neg_one_mask)
            x = -1.0f;

        float y = 0.0f;
        if (bit_index & fast_grad4::y_one_mask)
            y = 1.0f;
        if (bit_index & fast_grad4::y_neg_one_mask)
            y = -1.0f;

        float z = 0.0f;
        if (bit_index & fast_grad4::z_one_mask)
            z = 1.0f;
        if (bit_index & fast_grad4::z_neg_one_mask)
            z = -1.0f;

        float w = 0.0f;
        if (bit_index & fast_grad4::w_one_mask)
            w = 1.0f;
        if (bit_index & fast_grad4::w_neg_one_mask)
            w = -1.0f;

        return Vec4(x,y,z,w);
#endif
    }

    struct NoDerivs {
        static OSL_FORCEINLINE constexpr bool has_derivs() { return false; }
        static OSL_FORCEINLINE void set_dx(float) {}
        static OSL_FORCEINLINE void set_dy(float) {}
        static OSL_FORCEINLINE void set_dz(float) {}
        static OSL_FORCEINLINE void set_dw(float) {}
    };

    struct DxRef {
    private:
        float & m_dx;
    public:

        explicit OSL_FORCEINLINE DxRef(float &dx)
        : m_dx(dx)
        {}

        static OSL_FORCEINLINE constexpr bool has_derivs() { return true; }
        OSL_FORCEINLINE void set_dx(float val) { m_dx = val; }
    };

    struct DxDyRef {
    private:
        float & m_dx;
        float & m_dy;
    public:

        explicit OSL_FORCEINLINE DxDyRef(float &dx, float &dy)
        : m_dx(dx)
        , m_dy(dy)
        {}

        static OSL_FORCEINLINE constexpr bool has_derivs() { return true; }
        OSL_FORCEINLINE void set_dx(float val) { m_dx = val; }
        OSL_FORCEINLINE void set_dy(float val) { m_dy = val; }
    };

    struct DxDyDzRef {
    private:
        float & m_dx;
        float & m_dy;
        float & m_dz;
    public:

        explicit OSL_FORCEINLINE DxDyDzRef(float &dx, float &dy, float &dz)
        : m_dx(dx)
        , m_dy(dy)
        , m_dz(dz)
        {}

        static OSL_FORCEINLINE constexpr bool has_derivs() { return true; }
        OSL_FORCEINLINE void set_dx(float val) { m_dx = val; }
        OSL_FORCEINLINE void set_dy(float val) { m_dy = val; }
        OSL_FORCEINLINE void set_dz(float val) { m_dz = val; }
    };

    struct DxDyDzDwRef {
    private:
        float & m_dx;
        float & m_dy;
        float & m_dz;
        float & m_dw;
    public:

        explicit OSL_FORCEINLINE DxDyDzDwRef(float &dx, float &dy, float &dz, float &dw)
        : m_dx(dx)
        , m_dy(dy)
        , m_dz(dz)
        , m_dw(dw)
        {}

        static OSL_FORCEINLINE constexpr bool has_derivs() { return true; }
        OSL_FORCEINLINE void set_dx(float val) { m_dx = val; }
        OSL_FORCEINLINE void set_dy(float val) { m_dy = val; }
        OSL_FORCEINLINE void set_dz(float val) { m_dz = val; }
        OSL_FORCEINLINE void set_dw(float val) { m_dw = val; }
    };

    // 1D simplex noise with derivative.
    // If the last argument is not null, the analytic derivative
    // is also calculated.
    template<int seedT, typename OptionalDerivsT = NoDerivs>
    static OSL_FORCEINLINE float
    simplexnoise1 (float x, OptionalDerivsT derivPolicy = OptionalDerivsT())
    {
        int i0 = OIIO::ifloor(x);
        int i1 = i0 + 1;
        float x0 = x - i0;
        float x1 = x0 - 1.0f;

        float x20 = x0*x0;
        float t0 = 1.0f - x20;
        //  if(t0 < 0.0f) t0 = 0.0f; // Never happens for 1D: x0<=1 always
        float t20 = t0 * t0;
        float t40 = t20 * t20;
        float gx0 = grad1<seedT> (i0);
        float n0 = t40 * gx0 * x0;

        float x21 = x1*x1;
        float t1 = 1.0f - x21;
        //  if(t1 < 0.0f) t1 = 0.0f; // Never happens for 1D: |x1|<=1 always
        float t21 = t1 * t1;
        float t41 = t21 * t21;
        float gx1 = grad1<seedT> (i1);
        float n1 = t41 * gx1 * x1;

        // Sum up and scale the result.  The scale is empirical, to make it
        // cover [-1,1], and to make it approximately match the range of our
        // Perlin noise implementation.
        const float scale = 0.36f;

        if (derivPolicy.has_derivs()) {
            // Compute derivative according to:
            // *dnoise_dx = -8.0f * t20 * t0 * x0 * (gx0 * x0) + t40 * gx0;
            // *dnoise_dx += -8.0f * t21 * t1 * x1 * (gx1 * x1) + t41 * gx1;
            float dnoise_dx = t20 * t0 * gx0 * x20;
            dnoise_dx += t21 * t1 * gx1 * x21;
            dnoise_dx *= -8.0f;
            dnoise_dx += t40 * gx0 + t41 * gx1;
            dnoise_dx *= scale;
            derivPolicy.set_dx(dnoise_dx);
        }

        return scale * (n0 + n1);
    }

    // 2D simplex noise with derivatives.
    // If the last two arguments are not null, the analytic derivative
    // (the 2D gradient of the scalar noise field) is also calculated.
    template<int seedT, typename OptionalDerivsT = NoDerivs>
    static OSL_FORCEINLINE float
    simplexnoise2 (float x, float y, OptionalDerivsT derivPolicy = OptionalDerivsT())
    {
        // Skewing factors for 2D simplex grid:
        const float F2 = 0.366025403f;   // = 0.5*(sqrt(3.0)-1.0)
        const float G2 = 0.211324865f;  // = (3.0-Math.sqrt(3.0))/6.0

        /* Skew the input space to determine which simplex cell we're in */
        float s = ( x + y ) * F2; /* Hairy factor for 2D */
        float xs = x + s;
        float ys = y + s;
        int i = OIIO::ifloor(xs);
        int j = OIIO::ifloor(ys);

        float t = (float) (i + j) * G2;
        float X0 = i - t; /* Unskew the cell origin back to (x,y) space */
        float Y0 = j - t;
        float x0 = x - X0; /* The x,y distances from the cell origin */
        float y0 = y - Y0;

        /* For the 2D case, the simplex shape is an equilateral triangle.
         * Determine which simplex we are in. */
        //int i1, j1; // Offsets for second (middle) corner of simplex in (i,j) coords
        //if (x0 > y0) {
        //    i1 = 1; j1 = 0;   // lower triangle, XY order: (0,0)->(1,0)->(1,1)
        //} else {
        //    i1 = 0; j1 = 1;   // upper triangle, YX order: (0,0)->(0,1)->(1,1)
        //}
        int i1 = (x0 > y0);
        int j1 = !i1;

        // A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
        // a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
        // c = (3-sqrt(3))/6
        float x1 = x0 - i1 + G2; // Offsets for middle corner in (x,y) unskewed coords
        float y1 = y0 - j1 + G2;
        float x2 = x0 - 1.0f + 2.0f * G2; // Offsets for last corner in (x,y) unskewed coords
        float y2 = y0 - 1.0f + 2.0f * G2;


        // Noise contributions from the simplex corners
        // Calculate the contribution from the three corners

        // As we do not expect any coherency between data lanes
        // Hoisted work out of conditionals to encourage masking blending
        // versus a check for coherency
        // In other words we will do all the work, all the time versus
        // trying to manage it on a per lane basis.
        // NOTE: this may be slower if used for serial vs. simd
        float t0 = 0.5f - x0 * x0 - y0 * y0;
        float t20 = t0 * t0;
        float t40 = t20 * t20;
        Vec2 g0 = grad2<seedT> (i, j);
        float n0 = t40 * (g0.x * x0 + g0.y * y0);
        if (t0 < 0.0f)
            n0 = 0.0f;

        float t1 = 0.5f - x1 * x1 - y1 * y1;
        float t21 = t1 * t1;
        float t41 = t21 * t21;
        Vec2 g1 = grad2<seedT> (i+i1, j+j1);
        float n1 = t41 * (g1.x * x1 + g1.y * y1);
        if (t1 < 0.0f)
            n1 = 0.0f;

        float t2 = 0.5f - x2 * x2 - y2 * y2;
        float t22 = t2 * t2;
        float t42 = t22 * t22;
        Vec2 g2 = grad2<seedT> (i+1, j+1);
        float n2 = t42 * (g2.x * x2 + g2.y * y2);
        if (t2 < 0.0f)
            n2 = 0.0f;

        // Sum up and scale the result.  The scale is empirical, to make it
        // cover [-1,1], and to make it approximately match the range of our
        // Perlin noise implementation.
        const float scale = 64.0f;
        float noise = scale * (n0 + n1 + n2);

        // Compute derivative, if requested by supplying non-null pointers
        // for the last two arguments
        if (derivPolicy.has_derivs()) {
        /*  A straight, unoptimised calculation would be like:
         *    *dnoise_dx = -8.0f * t20 * t0 * x0 * ( g0.x * x0 + g0.y * y0 ) + t40 * g0.x;
         *    *dnoise_dy = -8.0f * t20 * t0 * y0 * ( g0.x * x0 + g0.y * y0 ) + t40 * g0.y;
         *    *dnoise_dx += -8.0f * t21 * t1 * x1 * ( g1.x * x1 + g1.y * y1 ) + t41 * g1.x;
         *    *dnoise_dy += -8.0f * t21 * t1 * y1 * ( g1.x * x1 + g1.y * y1 ) + t41 * g1.y;
         *    *dnoise_dx += -8.0f * t22 * t2 * x2 * ( g2.x * x2 + g2.y * y2 ) + t42 * g2.x;
         *    *dnoise_dy += -8.0f * t22 * t2 * y2 * ( g2.x * x2 + g2.y * y2 ) + t42 * g2.y;
         */
            // As we always calculated g# above,
            // we need to zero out those who were invalid
            // before they are used in more calculations
            if (t0 < 0.0f)
                g0 = Vec2(0.0f);
            float temp0 = t20 * t0 * (g0.x * x0 + g0.y * y0);
            float dnoise_dx = temp0 * x0;
            float dnoise_dy = temp0 * y0;
            if (t1 < 0.0f)
                g1 = Vec2(0.0f);
            float temp1 = t21 * t1 * (g1.x * x1 + g1.y * y1);
            dnoise_dx += temp1 * x1;
            dnoise_dy += temp1 * y1;
            if (t2 < 0.0f)
                g2 = Vec2(0.0f);
            float temp2 = t22 * t2 * (g2.x* x2 + g2.y * y2);
            dnoise_dx += temp2 * x2;
            dnoise_dy += temp2 * y2;
            dnoise_dx *= -8.0f;
            dnoise_dy *= -8.0f;
            dnoise_dx += t40 * g0.x + t41 * g1.x + t42 * g2.x;
            dnoise_dy += t40 * g0.y + t41 * g1.y + t42 * g2.y;
            dnoise_dx *= scale; /* Scale derivative to match the noise scaling */
            dnoise_dy *= scale;
            derivPolicy.set_dx(dnoise_dx);
            derivPolicy.set_dy(dnoise_dy);
        }
        return noise;
    }

    // 3D simplex noise with derivatives.
    // If the last tthree arguments are not null, the analytic derivative
    // (the 3D gradient of the scalar noise field) is also calculated.
    template<int seedT, typename OptionalDerivsT = NoDerivs>
    static OSL_FORCEINLINE float
    simplexnoise3 (float x, float y, float z, OptionalDerivsT derivPolicy = OptionalDerivsT())
    {
        // Skewing factors for 3D simplex grid:
        const float F3 = 0.333333333f;   // = 1/3
        const float G3 = 0.166666667f;   // = 1/6

        // Skew the input space to determine which simplex cell we're in
        float s = (x+y+z)*F3; // Very nice and simple skew factor for 3D
        float xs = x+s;
        float ys = y+s;
        float zs = z+s;

        int i = OIIO::ifloor(xs);
        int j = OIIO::ifloor(ys);
        int k = OIIO::ifloor(zs);

        float t = (float)(i+j+k)*G3;
        float X0 = i-t; // Unskew the cell origin back to (x,y,z) space
        float Y0 = j-t;
        float Z0 = k-t;

        float x0 = x-X0; // The x,y,z distances from the cell origin
        float y0 = y-Y0;
        float z0 = z-Z0;

        // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
        // Determine which simplex we are in.
        int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
        int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
        // NOTE:  The GLSL version of the flags produced different results
        // These flags are derived directly from the conditional logic
        // which is repeated in the verification code block following
        int bg0 = (x0 >= y0);
        int bg1 = (y0 >= z0);
        int bg2 = (x0 >= z0);
        int nbg0 = !bg0;
        int nbg1 = !bg1;
        int nbg2 = !bg2;
        i1 = bg0 & (bg1 | bg2);
        j1 = nbg0 & bg1;
        k1 =  nbg1 & ((bg0 & nbg2) | nbg0) ;
        i2 = bg0 | (bg1 & bg2);
        j2 = bg1 | nbg0;
        k2 = (bg0 & nbg1) | (nbg0 &(nbg1 | nbg2));

#ifdef OSL_VERIFY_SIMPLEX3  // Keep around to validate the bit logic above
        {
               if (x0>=y0) {
                        if (y0>=z0) {
                            OSL_VERIFY(i1==1);
                            OSL_VERIFY(j1==0);
                            OSL_VERIFY(k1==0);
                            OSL_VERIFY(i2==1);
                            OSL_VERIFY(j2==1);
                            OSL_VERIFY(k2==0);  /* X Y Z order */
                        } else if (x0>=z0) {
                            OSL_VERIFY(i1==1);
                            OSL_VERIFY(j1==0);
                            OSL_VERIFY(k1==0);
                            OSL_VERIFY(i2==1);
                            OSL_VERIFY(j2==0);
                            OSL_VERIFY(k2==1);  /* X Z Y order */
                        } else {
                            OSL_VERIFY(i1==0);
                            OSL_VERIFY(j1==0);
                            OSL_VERIFY(k1==1);
                            OSL_VERIFY(i2==1);
                            OSL_VERIFY(j2==0);
                            OSL_VERIFY(k2==1);  /* Z X Y order */
                        }
                    } else { // x0<y0
                        if (y0<z0) {
                            OSL_VERIFY(i1==0);
                            OSL_VERIFY(j1==0);
                            OSL_VERIFY(k1==1);
                            OSL_VERIFY(i2==0);
                            OSL_VERIFY(j2==1);
                            OSL_VERIFY(k2==1);  /* Z Y X order */
                        } else if (x0<z0) {
                            OSL_VERIFY(i1==0);
                            OSL_VERIFY(j1==1);
                            OSL_VERIFY(k1==0);
                            OSL_VERIFY(i2==0);
                            OSL_VERIFY(j2==1);
                            OSL_VERIFY(k2==1);  /* Y Z X order */
                        } else {
                            OSL_VERIFY(i1==0);
                            OSL_VERIFY(j1==1);
                            OSL_VERIFY(k1==0);
                            OSL_VERIFY(i2==1);
                            OSL_VERIFY(j2==1);
                            OSL_VERIFY(k2==0);  /* Y X Z order */
                        }
                    }
        }
    #endif

        // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
        // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
        // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z),
        // where c = 1/6.
        float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
        float y1 = y0 - j1 + G3;
        float z1 = z0 - k1 + G3;
        float x2 = x0 - i2 + 2.0f * G3; // Offsets for third corner in (x,y,z) coords
        float y2 = y0 - j2 + 2.0f * G3;
        float z2 = z0 - k2 + 2.0f * G3;
        float x3 = x0 - 1.0f + 3.0f * G3; // Offsets for last corner in (x,y,z) coords
        float y3 = y0 - 1.0f + 3.0f * G3;
        float z3 = z0 - 1.0f + 3.0f * G3;

        // As we do not expect any coherency between data lanes
        // Hoisted work out of conditionals to encourage masking blending
        // versus a check for coherency
        // In other words we will do all the work, all the time versus
        // trying to manage it on a per lane basis.
        // NOTE: this may be slower if used for serial vs. simd

        // Calculate the contribution from the four corners
        float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0;
        float t20 = t0 * t0;
        float t40 = t20 * t20;
        Vec3 g0 = grad3<seedT>(i, j, k);
        // NOTE: avoid array access of points, always use
        // the real data members to avoid aliasing issues
        //n0 = t40 * (g0.x * x0 + g0.y * y0 + g0.z * z0);
        float n0 = t40 * (g0.x * x0 + g0.y * y0 + g0.z * z0);
        if (t0 < 0.0f)
            n0 = 0.0f;

        float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1;
        float t21 = t1 * t1;
        float t41 = t21 * t21;
        Vec3 g1 = grad3<seedT>(i+i1, j+j1, k+k1);
        float n1 = t41 * (g1.x * x1 + g1.y * y1 + g1.z * z1);
        if (t1 < 0.0f)
            n1 = 0.0f;

        float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2;
        float t22 = t2 * t2;
        float t42 = t22 * t22;
        Vec3 g2 = grad3<seedT>(i+i2, j+j2, k+k2);
        float n2 = t42 * (g2.x * x2 + g2.y * y2 + g2.z * z2);
        if (t2 < 0.0f)
            n2 = 0.0f;

        float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3;
        float t23 = t3 * t3;
        float t43 = t23 * t23;
        Vec3 g3 = grad3<seedT>(i+1, j+1, k+1);
        float n3 = t43 * (g3.x * x3 + g3.y * y3 + g3.z * z3);
        if (t3 < 0.0f)
            n3 = 0.0f;

        // Sum up and scale the result.  The scale is empirical, to make it
        // cover [-1,1], and to make it approximately match the range of our
        // Perlin noise implementation.
        constexpr float scale = 68.0f;
        float noise = scale * (n0 + n1 + n2 + n3);


        if (derivPolicy.has_derivs()) {
            /*  A straight, unoptimized calculation would be like:
            *    *dnoise_dx = -8.0f * t20 * t0 * x0 * dot(g0.x, g0.y, g0.z, x0, y0, z0) + t40 * g0.x;
            *    *dnoise_dy = -8.0f * t20 * t0 * y0 * dot(g0.x, g0.y, g0.z, x0, y0, z0) + t40 * g0.y;
            *    *dnoise_dz = -8.0f * t20 * t0 * z0 * dot(g0.x, g0.y, g0.z, x0, y0, z0) + t40 * g0.z;
            *    *dnoise_dx += -8.0f * t21 * t1 * x1 * dot(g1.x, g1.y, g1.z, x1, y1, z1) + t41 * g1.x;
            *    *dnoise_dy += -8.0f * t21 * t1 * y1 * dot(g1.x, g1.y, g1.z, x1, y1, z1) + t41 * g1.y;
            *    *dnoise_dz += -8.0f * t21 * t1 * z1 * dot(g1.x, g1.y, g1.z, x1, y1, z1) + t41 * g1.z;
            *    *dnoise_dx += -8.0f * t22 * t2 * x2 * dot(g2.x, g2.y, g2.z, x2, y2, z2) + t42 * g2.x;
            *    *dnoise_dy += -8.0f * t22 * t2 * y2 * dot(g2.x, g2.y, g2.z, x2, y2, z2) + t42 * g2.y;
            *    *dnoise_dz += -8.0f * t22 * t2 * z2 * dot(g2.x, g2.y, g2.z, x2, y2, z2) + t42 * g2.z;
            *    *dnoise_dx += -8.0f * t23 * t3 * x3 * dot(g3.x, g3.y, g3.z, x3, y3, z3) + t43 * g3.x;
            *    *dnoise_dy += -8.0f * t23 * t3 * y3 * dot(g3.x, g3.y, g3.z, x3, y3, z3) + t43 * g3.y;
            *    *dnoise_dz += -8.0f * t23 * t3 * z3 * dot(g3.x, g3.y, g3.z, x3, y3, z3) + t43 * g3.z;
            */
            // As we always calculated g# above,
            // we need to zero out those who were invalid
            // before they are used in more calculations
            if (t0 < 0.0f)
                g0 = Vec3(0.0f);
            float temp0 = t20 * t0 * (g0.x * x0 + g0.y * y0 + g0.z * z0);
            float dnoise_dx = temp0 * x0;
            float dnoise_dy = temp0 * y0;
            float dnoise_dz = temp0 * z0;
            if (t1 < 0.0f)
                g1 = Vec3(0.0f);
            float temp1 = t21 * t1 * (g1.x * x1 + g1.y * y1 + g1.z * z1);
            dnoise_dx += temp1 * x1;
            dnoise_dy += temp1 * y1;
            dnoise_dz += temp1 * z1;
            if (t2 < 0.0f)
                g2 = Vec3(0.0f);
            float temp2 = t22 * t2 * (g2.x * x2 + g2.y * y2 + g2.z * z2);
            dnoise_dx += temp2 * x2;
            dnoise_dy += temp2 * y2;
            dnoise_dz += temp2 * z2;
            if (t3 < 0.0f)
                g3 = Vec3(0.0f);
            float temp3 = t23 * t3 * (g3.x * x3 + g3.y * y3 + g3.z * z3);
            dnoise_dx += temp3 * x3;
            dnoise_dy += temp3 * y3;
            dnoise_dz += temp3 * z3;
            dnoise_dx *= -8.0f;
            dnoise_dy *= -8.0f;
            dnoise_dz *= -8.0f;
            dnoise_dx += t40 * g0.x + t41 * g1.x + t42 * g2.x + t43 * g3.x;
            dnoise_dy += t40 * g0.y + t41 * g1.y + t42 * g2.y + t43 * g3.y;
            dnoise_dz += t40 * g0.z + t41 * g1.z + t42 * g2.z + t43 * g3.z;
            dnoise_dx *= scale; // Scale derivative to match the noise scaling
            dnoise_dy *= scale;
            dnoise_dz *= scale;

            derivPolicy.set_dx(dnoise_dx);
            derivPolicy.set_dy(dnoise_dy);
            derivPolicy.set_dz(dnoise_dz);
        }

        return noise;
    }


    // 4D simplex noise with derivatives.
    // If the last four arguments are not null, the analytic derivative
    // (the 4D gradient of the scalar noise field) is also calculated.
    template<int seedT, typename OptionalDerivsT = NoDerivs>
    static OSL_FORCEINLINE float
    simplexnoise4 (float x, float y, float z, float w, OptionalDerivsT derivPolicy = OptionalDerivsT())
    {
        // The skewing and unskewing factors are hairy again for the 4D case
        const float F4 = 0.309016994; // F4 = (Math.sqrt(5.0)-1.0)/4.0
        const float G4 = 0.138196601; // G4 = (5.0-Math.sqrt(5.0))/20.0

        // Gradients at simplex corners
        //const float *g0 = zero, *g1 = zero, *g2 = zero, *g3 = zero, *g4 = zero;

        // Noise contributions from the four simplex corners
        //float n0=0.0f, n1=0.0f, n2=0.0f, n3=0.0f, n4=0.0f;
        //float t20 = 0.0f, t21 = 0.0f, t22 = 0.0f, t23 = 0.0f, t24 = 0.0f;
        //float t40 = 0.0f, t41 = 0.0f, t42 = 0.0f, t43 = 0.0f, t44 = 0.0f;

        // Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
        float s = (x + y + z + w) * F4; // Factor for 4D skewing
        float xs = x + s;
        float ys = y + s;
        float zs = z + s;
        float ws = w + s;
        int i = OIIO::ifloor(xs);
        int j = OIIO::ifloor(ys);
        int k = OIIO::ifloor(zs);
        int l = OIIO::ifloor(ws);

        float t = (i + j + k + l) * G4; // Factor for 4D unskewing
        float X0 = i - t; // Unskew the cell origin back to (x,y,z,w) space
        float Y0 = j - t;
        float Z0 = k - t;
        float W0 = l - t;

        float x0 = x - X0;  // The x,y,z,w distances from the cell origin
        float y0 = y - Y0;
        float z0 = z - Z0;
        float w0 = w - W0;

        // For the 4D case, the simplex is a 4D shape I won't even try to describe.
        // To find out which of the 24 possible simplices we're in, we need to
        // determine the magnitude ordering of x0, y0, z0 and w0.
        // The method below is a reasonable way of finding the ordering of x,y,z,w
        // and then find the correct traversal order for the simplex we’re in.
        // First, six pair-wise comparisons are performed between each possible pair
        // of the four coordinates, and then the results are used to add up binary
        // bits for an integer index into a precomputed lookup table, simplex[].
        int c1 = (x0 > y0) ? 32 : 0;
        int c2 = (x0 > z0) ? 16 : 0;
        int c3 = (y0 > z0) ? 8 : 0;
        int c4 = (x0 > w0) ? 4 : 0;
        int c5 = (y0 > w0) ? 2 : 0;
        int c6 = (z0 > w0) ? 1 : 0;
        int c = c1 | c2 | c3 | c4 | c5 | c6; // '|' is mostly faster than '+'

        int i1, j1, k1, l1; // The integer offsets for the second simplex corner
        int i2, j2, k2, l2; // The integer offsets for the third simplex corner
        int i3, j3, k3, l3; // The integer offsets for the fourth simplex corner

        // simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
        // Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
        // impossible. Only the 24 indices which have non-zero entries make any sense.
        // We use a thresholding to set the coordinates in turn from the largest magnitude.
        // The number 3 in the "simplex" array is at the position of the largest coordinate.

#ifndef __OSL_LUT_SIMPLEX
    #define __OSL_LUT_SIMPLEX 0
#endif
#if __OSL_LUT_SIMPLEX
        // TODO: get rid of this lookup, try it with pure conditionals,
        // TODO: This should not be required, backport it from Bill's GLSL code!
        i1 = fast_simplex::lut[c][0]>=3 ? 1 : 0;
        j1 = fast_simplex::lut[c][1]>=3 ? 1 : 0;
        k1 = fast_simplex::lut[c][2]>=3 ? 1 : 0;
        l1 = fast_simplex::lut[c][3]>=3 ? 1 : 0;
        // The number 2 in the "simplex" array is at the second largest coordinate.
        i2 = fast_simplex::lut[c][0]>=2 ? 1 : 0;
        j2 = fast_simplex::lut[c][1]>=2 ? 1 : 0;
        k2 = fast_simplex::lut[c][2]>=2 ? 1 : 0;
        l2 = fast_simplex::lut[c][3]>=2 ? 1 : 0;
        // The number 1 in the "simplex" array is at the second smallest coordinate.
        i3 = fast_simplex::lut[c][0]>=1 ? 1 : 0;
        j3 = fast_simplex::lut[c][1]>=1 ? 1 : 0;
        k3 = fast_simplex::lut[c][2]>=1 ? 1 : 0;
        l3 = fast_simplex::lut[c][3]>=1 ? 1 : 0;
        // The fifth corner has all coordinate offsets = 1, so no need to look that up.
#else
        uint64_t bit_index = uint64_t(1u) << c;

        i1 = (bit_index & fast_simplex::corner2_i_offset_mask) >> c;
        j1 = (bit_index & fast_simplex::corner2_j_offset_mask) >> c;
        k1 = (bit_index & fast_simplex::corner2_k_offset_mask) >> c;
        l1 = (bit_index & fast_simplex::corner2_l_offset_mask) >> c;
        // The number 2 in the "simplex" array is at the second largest coordinate.
        i2 = (bit_index & fast_simplex::corner3_i_offset_mask) >> c;
        j2 = (bit_index & fast_simplex::corner3_j_offset_mask) >> c;
        k2 = (bit_index & fast_simplex::corner3_k_offset_mask) >> c;
        l2 = (bit_index & fast_simplex::corner3_l_offset_mask) >> c;
        // The number 1 in the "simplex" array is at the second smallest coordinate.
        i3 = (bit_index & fast_simplex::corner4_i_offset_mask) >> c;
        j3 = (bit_index & fast_simplex::corner4_j_offset_mask) >> c;
        k3 = (bit_index & fast_simplex::corner4_k_offset_mask) >> c;
        l3 = (bit_index & fast_simplex::corner4_l_offset_mask) >> c;
        // The fifth corner has all coordinate offsets = 1, so no need to look that up.
#endif


        float x1 = x0 - i1 + G4; // Offsets for second corner in (x,y,z,w) coords
        float y1 = y0 - j1 + G4;
        float z1 = z0 - k1 + G4;
        float w1 = w0 - l1 + G4;
        float x2 = x0 - i2 + 2.0f * G4; // Offsets for third corner in (x,y,z,w) coords
        float y2 = y0 - j2 + 2.0f * G4;
        float z2 = z0 - k2 + 2.0f * G4;
        float w2 = w0 - l2 + 2.0f * G4;
        float x3 = x0 - i3 + 3.0f * G4; // Offsets for fourth corner in (x,y,z,w) coords
        float y3 = y0 - j3 + 3.0f * G4;
        float z3 = z0 - k3 + 3.0f * G4;
        float w3 = w0 - l3 + 3.0f * G4;
        float x4 = x0 - 1.0f + 4.0f * G4; // Offsets for last corner in (x,y,z,w) coords
        float y4 = y0 - 1.0f + 4.0f * G4;
        float z4 = z0 - 1.0f + 4.0f * G4;
        float w4 = w0 - 1.0f + 4.0f * G4;

        // As we do not expect any coherency between data lanes
        // Hoisted work out of conditionals to encourage masking blending
        // versus a check for coherency
        // In other words we will do all the work, all the time versus
        // trying to manage it on a per lane basis.
        // NOTE: this may be slower if used for serial vs. simd

        // Calculate the contribution from the five corners
        // NOTE: avoid array access of points, always use
        // the real data members to avoid aliasing issues
        float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0 - w0*w0;
        float t20 = t0 * t0;
        float t40 = t20 * t20;
        Vec4 g0 = grad4<seedT>(i, j, k, l);
        float n0 = t40 * (g0.x * x0 + g0.y * y0 + g0.z * z0 + g0.w * w0);
        if (t0 < 0.0f)
            n0 = 0.0f;

        float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1 - w1*w1;
        float t21 = t1 * t1;
        float t41 = t21 * t21;
        Vec4 g1 = grad4<seedT>(i+i1, j+j1, k+k1, l+l1);
        float n1 = t41 * (g1.x * x1 + g1.y * y1 + g1.z * z1 + g1.w * w1);
        if (t1 < 0.0f)
            n1 = 0.0f;

        float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2 - w2*w2;
        float t22 = t2 * t2;
        float t42 = t22 * t22;
        Vec4 g2 = grad4<seedT>(i+i2, j+j2, k+k2, l+l2);
        float n2 = t42 * (g2.x * x2 + g2.y * y2 + g2.z * z2 + g2.w * w2);
        if (t2 < 0.0f)
            n2 = 0.0f;

        float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3 - w3*w3;
        float t23 = t3 * t3;
        float t43 = t23 * t23;
        Vec4 g3 = grad4<seedT>(i+i3, j+j3, k+k3, l+l3);
        float n3 = t43 * (g3.x * x3 + g3.y * y3 + g3.z * z3 + g3.w * w3);
        if (t3 < 0.0f)
            n3 = 0.0f;

        float t4 = 0.5f - x4*x4 - y4*y4 - z4*z4 - w4*w4;
        float t24 = t4 * t4;
        float t44 = t24 * t24;
        Vec4 g4 = grad4<seedT>(i+1, j+1, k+1, l+1);
        float n4 = t44 * (g4.x * x4 + g4.y * y4 + g4.z * z4 + g4.w * w4);
        if (t4 < 0.0f)
            n4 = 0.0f;

        // Sum up and scale the result.  The scale is empirical, to make it
        // cover [-1,1], and to make it approximately match the range of our
        // Perlin noise implementation.
        const float scale = 54.0f;
        float noise = scale * (n0 + n1 + n2 + n3 + n4);

        if (derivPolicy.has_derivs()) {
            /*  A straight, unoptimised calculation would be like:
            *    *dnoise_dx = -8.0f * t20 * t0 * x0 * dot(g0.x, g0.y, g0.z, g0.w, x0, y0, z0, w0) + t40 * g0.x;
            *    *dnoise_dy = -8.0f * t20 * t0 * y0 * dot(g0.x, g0.y, g0.z, g0.w, x0, y0, z0, w0) + t40 * g0.y;
            *    *dnoise_dz = -8.0f * t20 * t0 * z0 * dot(g0.x, g0.y, g0.z, g0.w, x0, y0, z0, w0) + t40 * g0.z;
            *    *dnoise_dw = -8.0f * t20 * t0 * w0 * dot(g0.x, g0.y, g0.z, g0.w, x0, y0, z0, w0) + t40 * g0.w;
            *    *dnoise_dx += -8.0f * t21 * t1 * x1 * dot(g1.x, g1.y, g1.z, g1.w, x1, y1, z1, w1) + t41 * g1.x;
            *    *dnoise_dy += -8.0f * t21 * t1 * y1 * dot(g1.x, g1.y, g1.z, g1.w, x1, y1, z1, w1) + t41 * g1.y;
            *    *dnoise_dz += -8.0f * t21 * t1 * z1 * dot(g1.x, g1.y, g1.z, g1.w, x1, y1, z1, w1) + t41 * g1.z;
            *    *dnoise_dw = -8.0f * t21 * t1 * w1 * dot(g1.x, g1.y, g1.z, g1.w, x1, y1, z1, w1) + t41 * g1.w;
            *    *dnoise_dx += -8.0f * t22 * t2 * x2 * dot(g2.x, g2.y, g2.z, g2.w, x2, y2, z2, w2) + t42 * g2.x;
            *    *dnoise_dy += -8.0f * t22 * t2 * y2 * dot(g2.x, g2.y, g2.z, g2.w, x2, y2, z2, w2) + t42 * g2.y;
            *    *dnoise_dz += -8.0f * t22 * t2 * z2 * dot(g2.x, g2.y, g2.z, g2.w, x2, y2, z2, w2) + t42 * g2.z;
            *    *dnoise_dw += -8.0f * t22 * t2 * w2 * dot(g2.x, g2.y, g2.z, g2.w, x2, y2, z2, w2) + t42 * g2.w;
            *    *dnoise_dx += -8.0f * t23 * t3 * x3 * dot(g3.x, g3.y, g3.z, g3.w, x3, y3, z3, w3) + t43 * g3.x;
            *    *dnoise_dy += -8.0f * t23 * t3 * y3 * dot(g3.x, g3.y, g3.z, g3.w, x3, y3, z3, w3) + t43 * g3.y;
            *    *dnoise_dz += -8.0f * t23 * t3 * z3 * dot(g3.x, g3.y, g3.z, g3.w, x3, y3, z3, w3) + t43 * g3.z;
            *    *dnoise_dw += -8.0f * t23 * t3 * w3 * dot(g3.x, g3.y, g3.z, g3.w, x3, y3, z3, w3) + t43 * g3.w;
            *    *dnoise_dx += -8.0f * t24 * t4 * x4 * dot(g4.x, g4.y, g4.z, g4.w, x4, y4, z4, w4) + t44 * g4.x;
            *    *dnoise_dy += -8.0f * t24 * t4 * y4 * dot(g4.x, g4.y, g4.z, g4.w, x4, y4, z4, w4) + t44 * g4.y;
            *    *dnoise_dz += -8.0f * t24 * t4 * z4 * dot(g4.x, g4.y, g4.z, g4.w, x4, y4, z4, w4) + t44 * g4.z;
            *    *dnoise_dw += -8.0f * t24 * t4 * w4 * dot(g4.x, g4.y, g4.z, g4.w, x4, y4, z4, w4) + t44 * g4.w;
            */
            // As we always calculated g# above,
            // we need to zero out those who were invalid
            // before they are used in more calculations
            if (t0 < 0.0f)
                g0 = Vec4(0.0f);
            float temp0 = t20 * t0 * (g0.x * x0 + g0.y * y0 + g0.z * z0 + g0.w * w0);
            float dnoise_dx = temp0 * x0;
            float dnoise_dy = temp0 * y0;
            float dnoise_dz = temp0 * z0;
            float dnoise_dw = temp0 * w0;
            if (t1 < 0.0f)
                g1 = Vec4(0.0f);
            float temp1 = t21 * t1 * (g1.x * x1 + g1.y * y1 + g1.z * z1 + g1.w * w1);
            dnoise_dx += temp1 * x1;
            dnoise_dy += temp1 * y1;
            dnoise_dz += temp1 * z1;
            dnoise_dw += temp1 * w1;
            if (t2 < 0.0f)
                g2 = Vec4(0.0f);
            float temp2 = t22 * t2 * (g2.x * x2 + g2.y * y2 + g2.z * z2 + g2.w * w2);
            dnoise_dx += temp2 * x2;
            dnoise_dy += temp2 * y2;
            dnoise_dz += temp2 * z2;
            dnoise_dw += temp2 * w2;
            if (t3 < 0.0f)
                g3 = Vec4(0.0f);
            float temp3 = t23 * t3 * (g3.x * x3 + g3.y * y3 + g3.z * z3 + g3.w * w3);
            dnoise_dx += temp3 * x3;
            dnoise_dy += temp3 * y3;
            dnoise_dz += temp3 * z3;
            dnoise_dw += temp3 * w3;
            if (t4 < 0.0f)
                g4 = Vec4(0.0f);
            float temp4 = t24 * t4 * (g4.x * x4 + g4.y * y4 + g4.z * z4 + g4.w * w4);
            dnoise_dx += temp4 * x4;
            dnoise_dy += temp4 * y4;
            dnoise_dz += temp4 * z4;
            dnoise_dw += temp4 * w4;
            dnoise_dx *= -8.0f;
            dnoise_dy *= -8.0f;
            dnoise_dz *= -8.0f;
            dnoise_dw *= -8.0f;
            dnoise_dx += t40 * g0.x + t41 * g1.x + t42 * g2.x + t43 * g3.x + t44 * g4.x;
            dnoise_dy += t40 * g0.y + t41 * g1.y + t42 * g2.y + t43 * g3.y + t44 * g4.y;
            dnoise_dz += t40 * g0.z + t41 * g1.z + t42 * g2.z + t43 * g3.z + t44 * g4.z;
            dnoise_dw += t40 * g0.w + t41 * g1.w + t42 * g2.w + t43 * g3.w + t44 * g4.w;
            // Scale derivative to match the noise scaling
            dnoise_dx *= scale;
            dnoise_dy *= scale;
            dnoise_dz *= scale;
            dnoise_dw *= scale;


            derivPolicy.set_dx(dnoise_dx);
            derivPolicy.set_dy(dnoise_dy);
            derivPolicy.set_dz(dnoise_dz);
            derivPolicy.set_dw(dnoise_dw);
        }
        return noise;
    }

} // namespace sfm


} // namespace __OSL_WIDE_PVT or pvt


OSL_NAMESPACE_EXIT
