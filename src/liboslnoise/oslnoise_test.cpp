/*
Copyright (c) 2016 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <iostream>

#include <OpenImageIO/unittest.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/timer.h>

#include <OSL/oslnoise.h>

using namespace OSL;
using namespace OSL::oslnoise;
using namespace OIIO;


static int iterations = 2000000;
static int ntrials = 5;
static bool verbose = false;
static bool make_images = false;
const int imgres = 64;
const float imgscale = 8.0;
const float eps = 0.001;   // Comparison threshold for results
const Vec3 veps (eps,eps,eps);



namespace std {   // hack!
inline float abs (const Vec3& a) {
    return std::max (std::max (abs(a[0]), abs(a[1])), abs(a[2]));
}
}


// Image test for visual check
#define MAKE_IMAGE(noisename)                                           \
    for (int outdim = 1; outdim <= 3; outdim += 2) {                    \
        for (int indim = 1; indim <= 4; ++indim) {                      \
            ImageSpec spec (imgres, imgres, outdim, TypeDesc::UINT8);   \
            ImageBuf img (spec);                                        \
            for (int y = 0; y < imgres; ++y) {                          \
                float t = float(y)/imgres * imgscale;                   \
                for (int x = 0; x < imgres; ++x) {                      \
                    float s = float(x)/imgres * imgscale;               \
                    Vec3 r;                                             \
                    if (outdim == 1) {                                  \
                        if (indim == 1)                                 \
                            r[0] = noisename(s);                        \
                        else if (indim == 2)                            \
                            r[0] = noisename(s,t);                      \
                        else if (indim == 3)                            \
                            r[0] = noisename(Vec3(s,t,1.0));            \
                        else                                            \
                            r[0] = noisename(Vec3(s,t,1.0),2.0);        \
                    } else {                                            \
                        if (indim == 1)                                 \
                            r = v ## noisename(s);                      \
                        else if (indim == 2)                            \
                            r = v ## noisename(s,t);                    \
                        else if (indim == 3)                            \
                            r = v ## noisename(Vec3(s,t,1.0));          \
                        else                                            \
                            r = v ## noisename(Vec3(s,t,1.0),2.0);      \
                    }                                                   \
                    img.setpixel (x, y, &r[0]);                         \
                }                                                       \
            }                                                           \
            img.write (Strutil::format ("osl_%s_%d_%d.tif", #noisename, outdim, indim)); \
        }                                                               \
    }



template <typename FUNC, typename T>
void benchmark1 (string_view funcname, FUNC func, T x)
{
    auto repeat_func = [&](){
        // Unroll the loop 8 times
        auto r = func(x); DoNotOptimize (r); clobber_all_memory();
        r = func(x); DoNotOptimize (r); clobber_all_memory();
        r = func(x); DoNotOptimize (r); clobber_all_memory();
        r = func(x); DoNotOptimize (r); clobber_all_memory();
        r = func(x); DoNotOptimize (r); clobber_all_memory();
        r = func(x); DoNotOptimize (r); clobber_all_memory();
        r = func(x); DoNotOptimize (r); clobber_all_memory();
        r = func(x); DoNotOptimize (r); clobber_all_memory();
    };
    float time = time_trial (repeat_func, ntrials, iterations/8);
    std::cout << Strutil::format ("  %s: %7.1f Mcalls/sec\n",
                                  funcname, (iterations/1.0e6)/time);
}



template <typename FUNC, typename S, typename T>
void benchmark2 (string_view funcname, FUNC func, S x, T y)
{
    auto repeat_func = [&](){
        // Unroll the loop 8 times
        auto r = func(x,y); DoNotOptimize (r); clobber_all_memory();
        r = func(x,y); DoNotOptimize (r); clobber_all_memory();
        r = func(x,y); DoNotOptimize (r); clobber_all_memory();
        r = func(x,y); DoNotOptimize (r); clobber_all_memory();
        r = func(x,y); DoNotOptimize (r); clobber_all_memory();
        r = func(x,y); DoNotOptimize (r); clobber_all_memory();
        r = func(x,y); DoNotOptimize (r); clobber_all_memory();
        r = func(x,y); DoNotOptimize (r); clobber_all_memory();
    };
    float time = time_trial (repeat_func, ntrials, iterations/8);
    std::cout << Strutil::format ("  %s: %7.1f Mcalls/sec\n",
                                  funcname, (iterations/1.0e6)/time);
}



void
test_perlin ()
{
    const int N = 4;           // Samples per unit of the grid
    static float results_1d[2*N+1] = {
        0, 0.316772, 0.3125, 0.0604248, 0, 0.211304, 0.5625, 0.467651, 0
    };
    static float results_2d[2*N+1] = {
        0, 0.0874839, 0, 0.219915, 0, -0.32144, 0.3308, 0.567027, 0
    };
    static float results_3d[2*N+1] = {
        0, -0.298763, 0.12275, 0.0533643, 0, -0.106914, -0.61375, -0.504605, 0
    };
    static float results_4d[2*N+1] = {
        0, 0.261616, 0.1043, -0.0740929, 0, -0.0687704, -0.1043, 0.0645373, 0
    };
    static Vec3 vresults_1d[2*N+1] = {
        Vec3(0,0,0), Vec3(0.316772,-0.416016,-0.267334), Vec3(0.3125,-0.75,-0.625),
        Vec3(0.0604248,-0.489258,-0.487061), Vec3(0,0,0), Vec3(0.211304,0.43103,0.351196),
        Vec3(0.5625,0.5625,0.1875), Vec3(0.467651,0.247925,-0.124878), Vec3(0,0,0)
    };
    static Vec3 vresults_2d[2*N+1] = {
        Vec3(0,0,0), Vec3(0.0874839,0.414739,0.0248834), Vec3(0,0.4962,-0.4962),
        Vec3(0.219915,0.414739,-0.546962), Vec3(0,0,0), Vec3(-0.32144,-0.290742,0.526898),
        Vec3(0.3308,0,0.4962), Vec3(0.567027,0.290742,0.14888), Vec3(0,-0,0)
    };
    static Vec3 vresults_3d[2*N+1] = {
        Vec3(0,0,0), Vec3(-0.298763,-0.18861,0.0753458), Vec3(0.12275,-0.491,0),
        Vec3(0.0533643,-0.476305,-0.303479), Vec3(0,0,0), Vec3(-0.106914,0.331779,0.196719),
        Vec3(-0.61375,0.36825,0), Vec3(-0.504605,0.0252173,0.29568), Vec3(0,0,0)
    };
    static Vec3 vresults_4d[2*N+1] = {
        Vec3(0,0,0), Vec3(0.261616,-0.0846003,0.353395), Vec3(0.1043,-0.1043,-0.1043),
        Vec3(-0.0740929,-0.168687,-0.333449), Vec3(0,0,0), Vec3(-0.0687704,0.0508762,0.224071),
        Vec3(-0.1043,-0.15645,0), Vec3(0.0645373,-0.14268,-0.229348), Vec3(0,0,0)
    };
    for (int i = 0; i <= 2*N; ++i) {
        // Signed perlin noise
        float x = float(i)/float(N);
        Vec3 p (x, x, x);
        float t = x;
        float s1 = snoise (x);
        float s2 = snoise (x, x);
        float s3 = snoise (p);
        float s4 = snoise (p, t);
        // std::cerr << s4 << ", ";
        OIIO_CHECK_EQUAL_THRESH (s1, results_1d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (s2, results_2d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (s3, results_3d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (s4, results_4d[i], eps);

        // Test that unsigned is the same, with adjusted range
        float n1 = noise (x);
        float n2 = noise (x, x);
        float n3 = noise (p);
        float n4 = noise (p, t);
        OIIO_CHECK_EQUAL_THRESH (n1, 0.5f+0.5f*s1, eps);
        OIIO_CHECK_EQUAL_THRESH (n2, 0.5f+0.5f*s2, eps);
        OIIO_CHECK_EQUAL_THRESH (n3, 0.5f+0.5f*s3, eps);
        OIIO_CHECK_EQUAL_THRESH (n4, 0.5f+0.5f*s4, eps);

        // Test vector variety
        Vec3 vs1 = vsnoise (x);
        Vec3 vs2 = vsnoise (x, x);
        Vec3 vs3 = vsnoise (p);
        Vec3 vs4 = vsnoise (p, t);
        // std::cerr << "Vec3" << vs2 << ",";
        // std::cerr << i << " " << vs1 << ' ' << vs2 << ' ' << vs3 << ' ' << vs4 << "\n";
        OIIO_CHECK_EQUAL_THRESH (vs1, vresults_1d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (vs2, vresults_2d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (vs3, vresults_3d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (vs4, vresults_4d[i], eps);
    }

    if (make_images) {
        MAKE_IMAGE (noise);
    }

    // Time trials
    benchmark1 ("snoise(f)      ", snoise<float>, 0.5f);
    benchmark2 ("snoise(f,f)    ", snoise<float,float>, 0.5f, 0.5f);
    benchmark1 ("snoise(v)      ", snoise<const Vec3&>, Vec3(0,0,0));
    benchmark2 ("snoise(v,f)    ", snoise<const Vec3&,float>, Vec3(0,0,0), 0.5f);

    benchmark1 ("vsnoise(f)     ", vsnoise<float>, 0.5f);
    benchmark2 ("vsnoise(f,f)   ", vsnoise<float,float>, 0.5f, 0.5f);
    benchmark1 ("vsnoise(v)     ", vsnoise<const Vec3&>, Vec3(0,0,0));
    benchmark2 ("vsnoise(v,f)   ", vsnoise<const Vec3&,float>, Vec3(0,0,0), 0.5f);
}



void
test_cell ()
{
    const int N = 1;
    static float results_1d[2*N+1] = {
        0.582426, 0.292355
    };
    static float results_2d[2*N+1] = {
        0.860313, 0.241469
    };
    static float results_3d[2*N+1] = {
        0.611068, 0.149855
    };
    static float results_4d[2*N+1] = {
        0.111573, 0.646492
    };
    static Vec3 vresults_1d[2*N+1] = {
        Vec3(0.860313,0.842521,0.974821), Vec3(0.295013,0.241469,0.0633514)
    };
    static Vec3 vresults_2d[2*N+1] = {
        Vec3(0.611068,0.185824,0.0413061), Vec3(0.73866,0.149855,0.984101)
    };
    static Vec3 vresults_3d[2*N+1] = {
        Vec3(0.111573,0.3251,0.339947), Vec3(0.576891,0.646492,0.270396)
    };
    static Vec3 vresults_4d[2*N+1] = {
        Vec3(0.14575,0.0431595,0.531032), Vec3(0.847126,0.733529,0.911791)
    };

    for (int i = 0; i < 2*N; ++i) {
        float x = 0.5f + float(i)/float(N);
        Vec3 p (x, x, x);
        float t = x;
        float s1 = cellnoise (x);
        float s2 = cellnoise (x, x);
        float s3 = cellnoise (p);
        float s4 = cellnoise (p, t);
        // std::cerr << s4 << ", ";
        OIIO_CHECK_EQUAL_THRESH (s1, results_1d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (s2, results_2d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (s3, results_3d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (s4, results_4d[i], eps);

        // Test vector variety
        Vec3 vs1 = vcellnoise (x);
        Vec3 vs2 = vcellnoise (x, x);
        Vec3 vs3 = vcellnoise (p);
        Vec3 vs4 = vcellnoise (p, t);
        // std::cerr << "Vec3" << vs4 << ",";
        OIIO_CHECK_EQUAL_THRESH (vs1, vresults_1d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (vs2, vresults_2d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (vs3, vresults_3d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (vs4, vresults_4d[i], eps);
    }

    if (make_images) {
        MAKE_IMAGE (cellnoise);
    }

    // Time trials
    benchmark1 ("cellnoise(f)   ", cellnoise<float>, 0.5f);
    benchmark2 ("cellnoise(f,f) ", cellnoise<float,float>, 0.5f, 0.5f);
    benchmark1 ("cellnoise(v)   ", cellnoise<const Vec3&>, Vec3(0,0,0));
    benchmark2 ("cellnoise(v,f) ", cellnoise<const Vec3&,float>, Vec3(0,0,0), 0.5f);

    benchmark1 ("vcellnoise(f)  ", vcellnoise<float>, 0.5f);
    benchmark2 ("vcellnoise(f,f)", vcellnoise<float,float>, 0.5f, 0.5f);
    benchmark1 ("vcellnoise(v)  ", vcellnoise<const Vec3&>, Vec3(0,0,0));
    benchmark2 ("vcellnoise(v,f)", vcellnoise<const Vec3&,float>, Vec3(0,0,0), 0.5f);
}



void
test_hash ()
{
    const int N = 1;
    static float results_1d[2*N+1] = {
        0.983315, 0.914303
    };
    static float results_2d[2*N+1] = {
        0.716016, 0.360764
    };
    static float results_3d[2*N+1] = {
        0.311887, 0.273953
    };
    static float results_4d[2*N+1] = {
        0.480826, 0.84257
    };
    static Vec3 vresults_1d[2*N+1] = {
        Vec3(0.253513, 0.658608, 0.718555), Vec3(0.673784, 0.825515, 0.827693)
    };
    static Vec3 vresults_2d[2*N+1] = {
        Vec3(0.172726, 0.779528, 0.780237), Vec3(0.431263, 0.443849, 0.592155)
    };
    static Vec3 vresults_3d[2*N+1] = {
        Vec3(0.59078, 0.79799, 0.0202766), Vec3(0.593547, 0.156555, 0.978352)
    };
    static Vec3 vresults_4d[2*N+1] = {
        Vec3(0.297735, 0.240833, 0.386421), Vec3(0.642915, 0.367309, 0.879035)
    };

    for (int i = 0; i < 2*N; ++i) {
        float x = 0.5f + float(i)/float(N);
        Vec3 p (x, x, x);
        float t = x;
        float s1 = hashnoise (x);
        float s2 = hashnoise (x, x);
        float s3 = hashnoise (p);
        float s4 = hashnoise (p, t);
        // std::cerr << s4 << ", ";
        OIIO_CHECK_EQUAL_THRESH (s1, results_1d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (s2, results_2d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (s3, results_3d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (s4, results_4d[i], eps);

        // Test vector variety
        Vec3 vs1 = vhashnoise (x);
        Vec3 vs2 = vhashnoise (x, x);
        Vec3 vs3 = vhashnoise (p);
        Vec3 vs4 = vhashnoise (p, t);
        // std::cerr << "Vec3" << vs4 << ",";
        OIIO_CHECK_EQUAL_THRESH (vs1, vresults_1d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (vs2, vresults_2d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (vs3, vresults_3d[i], eps);
        OIIO_CHECK_EQUAL_THRESH (vs4, vresults_4d[i], eps);
    }

    if (make_images) {
        MAKE_IMAGE (hashnoise);
    }

    // Time trials
    benchmark1 ("hashnoise(f)   ", hashnoise<float>, 0.5f);
    benchmark2 ("hashnoise(f,f) ", hashnoise<float,float>, 0.5f, 0.5f);
    benchmark1 ("hashnoise(v)   ", hashnoise<const Vec3&>, Vec3(0,0,0));
    benchmark2 ("hashnoise(v,f) ", hashnoise<const Vec3&,float>, Vec3(0,0,0), 0.5f);

    benchmark1 ("vhashnoise(f)  ", vhashnoise<float>, 0.5f);
    benchmark2 ("vhashnoise(f,f)", vhashnoise<float,float>, 0.5f, 0.5f);
    benchmark1 ("vhashnoise(v)  ", vhashnoise<const Vec3&>, Vec3(0,0,0));
    benchmark2 ("vhashnoise(v,f)", vhashnoise<const Vec3&,float>, Vec3(0,0,0), 0.5f);
}



static void
getargs (int argc, const char *argv[])
{
    bool help = false;
    OIIO::ArgParse ap;
    ap.options ("oslnoise_test  (" OSL_INTRO_STRING ")\n"
                "Usage:  oslnoise_test [options]",
                // "%*", parse_files, "",
                "--help", &help, "Print help message",
                "-v", &verbose, "Verbose mode",
                "--img", &make_images, "Make test images",
                "--iterations %d", &iterations,
                    ustring::format("Number of iterations (default: %d)", iterations).c_str(),
                "--trials %d", &ntrials, "Number of trials",
                NULL);
    if (ap.parse (argc, (const char**)argv) < 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage ();
        exit (EXIT_FAILURE);
    }
    if (help) {
        ap.usage ();
        exit (EXIT_FAILURE);
    }
}



int
main (int argc, char const *argv[])
{
    getargs (argc, argv);

    test_perlin ();
    test_cell ();
    test_hash ();

    return unit_test_failures;
}
