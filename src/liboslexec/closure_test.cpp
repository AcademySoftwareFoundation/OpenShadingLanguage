/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include <vector>
#include <iostream>

#include <OpenImageIO/dassert.h>

#include "oslconfig.h"
#include "oslclosure.h"

using namespace OSL;


#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include <boost/random.hpp>

namespace {

class MyClosure : public BSDFClosure {
    float m_f;

public:
    MyClosure (float f) : m_f (f) { }

    void print_on (std::ostream &out) const
    {
        out << "my (" << m_f << ")";
    }

    bool get_cone(const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        return false;
    }

    Color3 eval (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        return Color3 (0, 0, 0);
    }

    void sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval, Labels &labels) const
    {
        pdf = 0, omega_in.setValue(0, 0, 0), eval.setValue(0, 0, 0);
    }

    float pdf (const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        return 0;
    }
};

ClosureColor create_component(const Color3& w, float f) {
    ClosureColor c;
    char* mem = c.allocate_component (sizeof (MyClosure));
    new (mem) MyClosure(f);
    c *= w;
    return c;
}


} // anonymous namespace

BOOST_AUTO_TEST_CASE (closure_test_add)
{
#if BOOST_VERSION >= 103900
   // avoid warnings from boost headers
    BOOST_CHECK_CLOSE(0.0f, 0.0f, 0.001f);
    BOOST_CHECK_SMALL(0.0f, 0.001f);
#endif
   // Create a closure with one component
    ClosureColor c;
    c += create_component (Color3(.1, .1, .1), 0.33f);

    BOOST_CHECK_EQUAL (c.ncomponents(), 1);

    // Add another component with different params.  It should now look
    // like two components, not combine with the others.
    c += create_component (Color3(.4, .4, .4), 0.5f);

    BOOST_CHECK_EQUAL (c.ncomponents(), 2);
    BOOST_CHECK_EQUAL (c.weight(1), Color3 (0.4, 0.4, 0.4));
    std::cout << "c = " << c << "\n";
}
