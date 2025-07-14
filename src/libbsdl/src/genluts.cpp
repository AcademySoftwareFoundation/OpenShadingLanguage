// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#define BAKE_BSDL_TABLES 1

#define BSDL_UNROLL()  // Do nothing

#include <BSDL/config.h>
using BSDLConfig = bsdl::BSDLDefaultConfig;

#include <BSDL/MTX/bsdf_dielectric_impl.h>
#include <BSDL/SPI/bsdf_backscatter_impl.h>
#include <BSDL/SPI/bsdf_clearcoat_impl.h>
#include <BSDL/SPI/bsdf_dielectric_impl.h>
#include <BSDL/SPI/bsdf_sheenltc_impl.h>
#include <BSDL/SPI/bsdf_thinlayer_impl.h>
#include <BSDL/bsdf_impl.h>
#include <BSDL/microfacet_tools_impl.h>

#include "parallel.h"

#include <algorithm>
#include <cstdio>
#include <tuple>
#include <vector>

#define BAKE_BSDF_LIST(E)       \
    E(spi::MiniMicrofacetGGX)   \
    E(spi::PlasticGGX)          \
    E(spi::DielectricFront)     \
    E(spi::DielectricBack)      \
    E(spi::CharlieSheen)        \
    E(spi::SheenLTC)            \
    E(spi::Thinlayer)           \
    E(mtx::DielectricReflFront) \
    E(mtx::DielectricBothFront) \
    E(mtx::DielectricBothBack)

#define LUT_PLACEHOLDER(type)                           \
    BSDL_ENTER_NAMESPACE                                \
    BSDL_INLINE_METHOD type::Energy& type::get_energy() \
    {                                                   \
        static Energy energy = {};                      \
        return energy;                                  \
    }                                                   \
    BSDL_LEAVE_NAMESPACE

BAKE_BSDF_LIST(LUT_PLACEHOLDER)

using namespace bsdl;

BSDL_INLINE uint32_t
ri_LP(uint32_t i)
{
    uint32_t r = 0;
    for (uint32_t v = 1U << 31; i; i >>= 1, v |= v >> 1)
        if (i & 1)
            r ^= v;
    return r;
}

BSDL_INLINE uint32_t
ri_LP_inv(uint32_t i)
{
    uint32_t r = 0;
    for (uint32_t v = 3U << 30; i; i >>= 1, v >>= 1)
        if (i & 1)
            r ^= v;
    return r;
}

BSDL_INLINE Imath::V3f
get_sample(int si, int AA, uint32_t scramble_x, uint32_t scramble_y,
           uint32_t scramble_z)
{
    const uint32_t ex = si % AA;
    const uint32_t ey = si / AA;

    const uint32_t upper   = (ex ^ (scramble_x >> 16)) << 16;
    const uint32_t lpUpper = ri_LP(upper) ^ scramble_y;
    const uint32_t delta   = (ey << 16) ^ (lpUpper & 0xFFFF0000u);
    const uint32_t lower   = ri_LP_inv(delta);
    const uint32_t index   = upper | lower;
    const uint32_t x       = index ^ scramble_x;
    const uint32_t y       = lpUpper ^ delta;
    const float jx         = (x & 65535) * (1 / 65536.0f);
    const float jy         = (y & 65535) * (1 / 65536.0f);
    uint32_t rz = scramble_z, ii = index;
    for (uint64_t v2 = uint64_t(3) << 62; ii; ii >>= 1, v2 ^= v2 >> 1)
        if (ii & 1)
            rz ^= uint32_t(v2 >> 31);

    Imath::V3f v = { (ex + jx) / AA, (ey + jy) / AA, rz * 2.3283063e-10f };
    assert(v.x >= 0);
    assert(v.x < 1);
    assert(v.y >= 0);
    assert(v.y < 1);
    assert(v.z >= 0);
    assert(v.z < 1);
    return v;
}

template<typename BSDF>
BSDL_INLINE float
compute_E(float cos_theta, const BSDF& bsdf, uint32_t fresnel_index,
          uint32_t roughness_index)
{
    auto fasthash64_mix = [](uint64_t h) -> uint64_t {
        h ^= h >> 23;
        h *= 0x2127599bf4325c37ULL;
        h ^= h >> 47;
        return h;
    };
    auto fasthash64 =
        [&](const std::initializer_list<uint64_t> buf) -> uint64_t {
        const uint64_t m = 0x880355f21e6d1965ULL;
        uint64_t h       = (buf.size() * sizeof(uint64_t)) * m;
        for (const uint64_t v : buf) {
            h ^= fasthash64_mix(v);
            h *= m;
        }
        return fasthash64_mix(h);
    };
    auto randhash3 = [&](uint32_t x, uint32_t y, uint32_t z) -> uint32_t {
        return fasthash64({ (uint64_t(x) << 32) + y, uint64_t(z) });
    };
    constexpr int AA          = 128;
    constexpr int NUM_SAMPLES = AA * AA;
    assert(cos_theta > 0);
    const Imath::V3f wo = { sqrtf(1 - SQR(cos_theta)), 0, cos_theta };
    float E             = 0;
    uint32_t seedx      = randhash3(fresnel_index, roughness_index, 0);
    uint32_t seedy      = randhash3(fresnel_index, roughness_index, 1);
    uint32_t seedz      = randhash3(fresnel_index, roughness_index, 2);
    for (int i = 0; i < NUM_SAMPLES; i++) {
        Imath::V3f rnd = get_sample(i, AA, seedx, seedy, seedz);
        float out      = bsdf.sample(wo, rnd.x, rnd.y, rnd.z).weight.max();

        // accumulate result progressively to minimize error with large sample counts
        E = LERP(1.0f / (1.0f + i), E, out);
    }
    assert(E >= 0);
    assert(E <= 1.01f);  // Allow for some error
    return std::min(E, 1.0f);
}

template<typename BSDF>
BSDL_INLINE void
bake_emiss_tables(const std::string& output_dir)
{
    float* storedE = BSDF::get_energy().data;

    printf("Generating LUTs for %s ...\n", BSDF::struct_name());

    parallel_for(0, BSDF::Nf, [&](unsigned f) {
        const float fresnel_index = float(f)
                                    * (1.0f / std::max(1, BSDF::Nf - 1));
        for (int r = 0; r < BSDF::Nr; r++) {
            const float roughness_index
                = BSDF::Nr > 1 ? float(r) * (1.0f / (BSDF::Nr - 1)) : 0.0f;
            for (int c = 0; c < BSDF::Nc; c++) {
                int idx = f * BSDF::Nr * BSDF::Nc + r * BSDF::Nc + c;
                const BSDF bsdf(BSDF::get_cosine(c), roughness_index,
                                fresnel_index);
                storedE[idx] = 1 - compute_E(BSDF::get_cosine(c), bsdf, f, r);
            }
        }
    });

    std::string out_file = output_dir + "/" + BSDF::lut_header();
    FILE* outf           = fopen(out_file.c_str(), "wb");
    if (!outf) {
        printf("Failed to open %s for writing\n", out_file.c_str());
        exit(-1);
    }

    fprintf(outf, "#pragma once\n\n");
    fprintf(outf, "BSDL_ENTER_NAMESPACE\n\n");
    fprintf(outf, "namespace %s {\n\n", BSDF::NS);
    fprintf(outf, "BSDL_INLINE_METHOD %s::Energy& %s::get_energy()\n",
            BSDF::struct_name(), BSDF::struct_name());
    fprintf(outf, "{\n");
    fprintf(outf, "    static Energy energy = {{\n");
    for (int f = 0, idx = 0; f < BSDF::Nf; f++) {
        for (int r = 0; r < BSDF::Nr; r++) {
            fprintf(outf, "       ");
            for (int c = 0; c < BSDF::Nc; c++, idx++)
                if (storedE[idx] == int(storedE[idx]))
                    fprintf(outf, " %12d.0f,", int(storedE[idx]));
                else
                    fprintf(outf, " %14.9gf,", storedE[idx]);
            fprintf(outf, "\n");
        }
    }
    fprintf(outf, "    }};\n");
    fprintf(outf, "    return energy;\n");
    fprintf(outf, "}\n\n");
    fprintf(outf, "} // namespace %s \n\n", BSDF::NS);
    fprintf(outf, "BSDL_LEAVE_NAMESPACE\n");
    fclose(outf);
    printf("Wrote LUTs to %s\n", out_file.c_str());
}

int
main(int argc, const char** argv)
{
    if (argc < 2) {
        printf("Must provide output dir for headers\n");
        return -1;
    }

#define DECLARE_DUMMY(type) type(0, 0, 0),

    std::tuple bsdf_list { BAKE_BSDF_LIST(DECLARE_DUMMY) };

    std::apply(
        [&](auto... args) {
            (bake_emiss_tables<
                 typename std::remove_reference<decltype(args)>::type>(argv[1]),
             ...);
        },
        bsdf_list);

    return 0;
}
