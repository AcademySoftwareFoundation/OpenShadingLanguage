/*
Copyright (c) 2017 Sony Pictures Imageworks Inc., et al.
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

#include <type_traits>

#include <OpenImageIO/dassert.h>
#include <OpenImageIO/unittest.h>

#include <OSL/oslconfig.h>
#include <OSL/dual.h>
#include <OSL/dual_vec.h>

using namespace OSL;

// Fix namespace problem with OIIO unit tests in older versions.
#if OIIO_VERSION < 10901
using namespace OIIO;
#endif

typedef Dual<float,1> Dualf;
typedef Dual2<float> Dual2f;



template<class T>
T crazy (const T& x)
{
    using namespace std;
    using namespace OSL;
    return x * x * cos(x);
}



template<class T>
T crazy (const T& x, const T& y)
{
    using namespace std;
    using namespace OSL;
    return x * y * exp(x+y);
}



// Compare 1D derivs with Duals to numerical differentiation. This sure
// isn't comprehensive, but it tests the general scheme.
void
test_derivs1 () {
    const float eps = 1e-3;
    static float domain[] = { 0, 0.1, 0.25, 0.333333, 0.91 };
    for (auto x : domain) {
        float f = crazy(x);
        float df_numeric = (crazy(x+eps) - crazy(x-eps)) / (2.0f*eps);
        Dual<float> xd (x, 1.0f);
        Dual<float> fd = crazy (xd);
        std::cout << "crazy(" << x << ")=" << f << ", numeric derivs ~= "
                 << df_numeric << ", Dual f = " << fd << "\n";

        OIIO_CHECK_EQUAL_THRESH (f, fd.val(), eps);
        OIIO_CHECK_EQUAL_THRESH (df_numeric, fd.dx(), eps);
    }
}



// Compare 2D derivs with Duals to numerical differentiation. This sure
// isn't comprehensive, but it tests the general scheme.
void
test_derivs2 () {
    const float eps = 1e-3;
    static float domain[] = { 0, 0.1, 0.25, 0.333333, 0.91 };
    for (auto x : domain) {
        for (auto y : domain) {
            float f = crazy(x,y);
            float df_dx_numeric = (crazy(x+eps,y) - crazy(x-eps,y)) / (2.0f*eps);
            float df_dy_numeric = (crazy(x,y+eps) - crazy(x,y-eps)) / (2.0f*eps);
            Dual<float,2> xd (x, 1.0f, 0.0f);
            Dual<float,2> yd (y, 0.0f, 1.0f);
            Dual<float,2> fd = crazy (xd, yd);
            std::cout << "crazy(" << x << "," << y << ")=" << f
                     << ", numeric derivs df/dx ~= "
                     << df_dx_numeric << ", df/dy ~= " << df_dy_numeric
                     << ", Dual f = " << fd << "\n";

            OIIO_CHECK_EQUAL_THRESH (f, fd.val(), eps);
            OIIO_CHECK_EQUAL_THRESH (df_dx_numeric, fd.dx(), eps);
            OIIO_CHECK_EQUAL_THRESH (df_dy_numeric, fd.dy(), eps);
        }
    }
}



void
test_metaprogramming ()
{
    // Does is_Dual<> correctly discern duals from non-duals?
    OIIO_CHECK_ASSERT (is_Dual<Dualf>());
    OIIO_CHECK_ASSERT (is_Dual<Dual2f>());
    OIIO_CHECK_ASSERT (! is_Dual<float>());
    OIIO_CHECK_ASSERT (! is_Dual<Vec3>());

    // Does Dualify<> correctly turn a scalar into a dual?
    OIIO_CHECK_ASSERT ((std::is_same<Dualify<float>::type, Dualf>::value));
    OIIO_CHECK_ASSERT ((std::is_same<Dualify<float,2>::type, Dual2f>::value));
    OIIO_CHECK_ASSERT ((std::is_same<Dualify<Dualf,1>::type, Dualf>::value));
    OIIO_CHECK_ASSERT ((std::is_same<Dualify<Dual2f,2>::type, Dual2f>::value));

    OIIO_CHECK_ASSERT ((std::is_same<UnDual<float >::type, float>::value));
    OIIO_CHECK_ASSERT ((std::is_same<UnDual<Vec3  >::type, Vec3> ::value));
    OIIO_CHECK_ASSERT ((std::is_same<UnDual<Dualf >::type, float>::value));
    OIIO_CHECK_ASSERT ((std::is_same<UnDual<Dual2f>::type, float>::value));
}



int main(int argc, char *argv[])
{
    test_metaprogramming ();
    test_derivs1 ();
    test_derivs2 ();

    // FIXME: Some day, expand to more exhaustive tests of Dual

    return 0;
}
