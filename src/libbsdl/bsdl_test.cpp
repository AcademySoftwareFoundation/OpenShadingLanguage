// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#define BSDL_UNROLL()

#include <BSDL/config.h>

using BSDLConfig = bsdl::BSDLDefaultConfig;

#include <BSDL/MTX/bsdf_schlick_impl.h>

#include <OpenImageIO/unittest.h>

using namespace bsdl;



struct TestDielectricBSDF : mtx::DielectricBSDF<mtx::SchlickFresnel> {
    using Base = mtx::DielectricBSDF<mtx::SchlickFresnel>;
    using Base::Base;
    using Base::reflection_probability;
};



static void
test_colored_schlick_sampling()
{
    const float rgb[] = { 0.1f, 0.2f, 0.8f };
    const Power F([&](int i) { return rgb[i]; }, 0.0f);
    const mtx::SchlickFresnel fresnel(F, F, 5.0f, 1.5f, false);
    const GGXDist dist(0.25f, 0.0f);
    const Imath::V3f wo(0.0f, 0.0f, 1.0f);
    const TestDielectricBSDF bsdf(dist, fresnel, wo.z, 0.25f, true, 0.0f);

    const float probability = (rgb[0] + rgb[1] + rgb[2]) / 3.0f;
    OIIO_CHECK_EQUAL_THRESH(bsdf.reflection_probability(F), probability, 1e-6f);
    OIIO_CHECK_EQUAL_THRESH(bsdf.reflection_probability(Power::UNIT()), 1.0f,
                            1e-6f);

    const Sample reflected   = bsdf.sample(wo, 0.5f, 0.5f, probability - 0.01f);
    const Sample transmitted = bsdf.sample(wo, 0.5f, 0.5f, probability + 0.01f);
    OIIO_CHECK_ASSERT(reflected.wi.z > 0.0f);
    OIIO_CHECK_ASSERT(transmitted.wi.z < 0.0f);

    const float rgb2[] = { 0.2f, 0.4f, 0.6f };
    const Power F2([&](int i) { return rgb2[i]; }, 0.0f);
    const mtx::SchlickFresnel fresnel2(F2, F2, 5.0f, 1.5f, false);
    const TestDielectricBSDF bsdf2(dist, fresnel2, wo.z, 0.25f, true, 0.0f);
    const float probability2 = (rgb2[0] + rgb2[1] + rgb2[2]) / 3.0f;

    const Sample reflected2   = bsdf2.sample(wo, 0.5f, 0.5f, 0.3f);
    const Sample transmitted2 = bsdf2.sample(wo, 0.5f, 0.5f, 0.8f);
    const Sample reflected1   = bsdf.sample(wo, 0.5f, 0.5f, 0.3f);
    const Sample transmitted1 = bsdf.sample(wo, 0.5f, 0.5f, 0.8f);
    OIIO_CHECK_EQUAL_THRESH(reflected1.pdf / reflected2.pdf,
                            probability / probability2, 1e-6f);
    const float transmission_probability  = 1.0f - probability;
    const float transmission_probability2 = 1.0f - probability2;
    OIIO_CHECK_EQUAL_THRESH(transmitted1.pdf / transmitted2.pdf,
                            transmission_probability
                                / transmission_probability2,
                            1e-6f);
    for (int i = 0; i < 3; ++i) {
        OIIO_CHECK_EQUAL_THRESH(reflected1.weight[i] / reflected2.weight[i],
                                (rgb[i] * probability2)
                                    / (rgb2[i] * probability),
                                1e-6f);
        OIIO_CHECK_EQUAL_THRESH(transmitted1.weight[i] / transmitted2.weight[i],
                                ((1.0f - rgb[i]) * transmission_probability2)
                                    / ((1.0f - rgb2[i])
                                       * transmission_probability),
                                1e-6f);
    }

    const float spectral[] = { 0.2f, 0.4f, 0.6f, 0.8f };
    const Power Fs([&](int i) { return spectral[i]; }, 500.0f);
    const mtx::SchlickFresnel spectral_fresnel(Fs, Fs, 5.0f, 1.5f, false);
    const TestDielectricBSDF spectral_bsdf(dist, spectral_fresnel, wo.z, 0.25f,
                                           true, 500.0f);
    OIIO_CHECK_EQUAL_THRESH(spectral_bsdf.reflection_probability(Fs), 0.5f,
                            1e-6f);
}



int
main(int /*argc*/, char* /*argv*/[])
{
    test_colored_schlick_sampling();
    return unit_test_failures;
}
