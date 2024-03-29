// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <type_traits>

#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/oslconfig.h>

#include <OpenImageIO/benchmark.h>
#include <OpenImageIO/unittest.h>

using namespace OSL;

typedef Dual<float, 1> Dualf;
typedef Dual2<float> Dual2f;



template<class T>
T
crazy(const T& x)
{
    using namespace std;
    using namespace OSL;
    return x * x * cos(x);
}



template<class T>
T
crazy(const T& x, const T& y)
{
    using namespace std;
    using namespace OSL;
    return x * y * exp(x + y);
}



// Compare 1D derivs with Duals to numerical differentiation. This sure
// isn't comprehensive, but it tests the general scheme.
void
test_derivs1()
{
    const float eps       = 1e-3;
    static float domain[] = { 0, 0.1, 0.25, 0.333333, 0.91 };
    for (auto x : domain) {
        float f          = crazy(x);
        float df_numeric = (crazy(x + eps) - crazy(x - eps)) / (2.0f * eps);
        Dual<float> xd(x, 1.0f);
        Dual<float> fd = crazy(xd);
        std::cout << "crazy(" << x << ")=" << f
                  << ", numeric derivs ~= " << df_numeric << ", Dual f = " << fd
                  << "\n";

        OIIO_CHECK_EQUAL_THRESH(f, fd.val(), eps);
        OIIO_CHECK_EQUAL_THRESH(df_numeric, fd.dx(), eps);
    }
}



// Compare 2D derivs with Duals to numerical differentiation. This sure
// isn't comprehensive, but it tests the general scheme.
void
test_derivs2()
{
    const float eps       = 1e-3;
    static float domain[] = { 0, 0.1, 0.25, 0.333333, 0.91 };
    for (auto x : domain) {
        for (auto y : domain) {
            float f             = crazy(x, y);
            float df_dx_numeric = (crazy(x + eps, y) - crazy(x - eps, y))
                                  / (2.0f * eps);
            float df_dy_numeric = (crazy(x, y + eps) - crazy(x, y - eps))
                                  / (2.0f * eps);
            Dual<float, 2> xd(x, 1.0f, 0.0f);
            Dual<float, 2> yd(y, 0.0f, 1.0f);
            Dual<float, 2> fd = crazy(xd, yd);
            std::cout << "crazy(" << x << "," << y << ")=" << f
                      << ", numeric derivs df/dx ~= " << df_dx_numeric
                      << ", df/dy ~= " << df_dy_numeric << ", Dual f = " << fd
                      << "\n";

            OIIO_CHECK_EQUAL_THRESH(f, fd.val(), eps);
            OIIO_CHECK_EQUAL_THRESH(df_dx_numeric, fd.dx(), eps);
            OIIO_CHECK_EQUAL_THRESH(df_dy_numeric, fd.dy(), eps);
        }
    }
}



void
test_metaprogramming()
{
    // Does is_Dual<> correctly discern duals from non-duals?
    OIIO_CHECK_ASSERT(is_Dual<Dualf>());
    OIIO_CHECK_ASSERT(is_Dual<Dual2f>());
    OIIO_CHECK_ASSERT(!is_Dual<float>());
    OIIO_CHECK_ASSERT(!is_Dual<Vec3>());

    // Does Dualify<> correctly turn a scalar into a dual?
    OIIO_CHECK_ASSERT((std::is_same<Dualify<float>::type, Dualf>::value));
    OIIO_CHECK_ASSERT((std::is_same<Dualify<float, 2>::type, Dual2f>::value));
    OIIO_CHECK_ASSERT((std::is_same<Dualify<Dualf, 1>::type, Dualf>::value));
    OIIO_CHECK_ASSERT((std::is_same<Dualify<Dual2f, 2>::type, Dual2f>::value));

    OIIO_CHECK_ASSERT((std::is_same<UnDual<float>::type, float>::value));
    OIIO_CHECK_ASSERT((std::is_same<UnDual<Vec3>::type, Vec3>::value));
    OIIO_CHECK_ASSERT((std::is_same<UnDual<Dualf>::type, float>::value));
    OIIO_CHECK_ASSERT((std::is_same<UnDual<Dual2f>::type, float>::value));
}



int
main(int /*argc*/, char* /*argv*/[])
{
    test_metaprogramming();
    test_derivs1();
    test_derivs2();

    // Some benchmarking
    std::cout << "\nBenchmarks:\n";
    using namespace OIIO;
    Benchmarker bench;
    Dual2f v(1.5f, 0.01f, 0.01f);
    clobber(v);
    bench(
        "-Dual2f", [&](const Dual2f& v) { return DoNotOptimize(-v); }, v);
    bench(
        "fast_neg(Dual2f)",
        [&](const Dual2f& v) { return DoNotOptimize(fast_neg(v)); }, v);
    bench(
        "log2(Dual2f)",
        [&](const Dual2f& v) { return DoNotOptimize(fast_log2(v)); }, v);

    // FIXME: Some day, expand to more exhaustive tests of Dual

    return unit_test_failures;
}
