/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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
#include "genclosure.h"
#include "oslexec_pvt.h"

using namespace OSL;

#include <boost/random.hpp>

#define MY_ID NBUILTIN_CLOSURES

namespace {

class MyClosure : public BSDFClosure {
    float m_f;

public:
    MyClosure (float f) : BSDFClosure(Labels::NONE, None), m_f (f) { }

    void setup() {};

    bool mergeable (const ClosurePrimitive *other) const {
        const MyClosure *comp = (const MyClosure *)other;
        return m_f == comp->m_f && BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "my"; }

    void print_on (std::ostream &out) const
    {
        out << "my (" << m_f << ")";
    }

    float albedo (const Vec3 &omega_out) const
    {
        return 1.0f;
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        return Color3 (0, 0, 0);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        pdf = 0, omega_in.setValue(0, 0, 0), eval.setValue(0, 0, 0);
        return Labels::NONE;
    }
};

#if 1

ClosureColor *create_component(ShadingContext *context, const Color3& w, float f) {
    ClosureComponent *comp = context->closure_component_allot(MY_ID, sizeof (MyClosure), 0);
    new (comp->mem) MyClosure(f);
    return context->closure_mul_allot(w, comp);
}


} // anonymous namespace

static bool my_compare(int id, const void *dataA, const void *dataB)
{
   ClosurePrimitive *primA = (ClosurePrimitive *)dataA;
   ClosurePrimitive *primB = (ClosurePrimitive *)dataB;
   return primA->mergeable (primB);
}

int main()
{
    ShadingSystemImpl *shadingsys = new ShadingSystemImpl();
    ShadingContext *context = shadingsys->get_context();

    ClosureParam my_params[] = { CLOSURE_FINISH_PARAM(MyClosure) };
    shadingsys->register_closure("my", MY_ID, my_params, sizeof(MyClosure), NULL, NULL, my_compare);
    ClosureColor *A = create_component (context, Color3(.1, .1, .1), 0.33f);
    // Add another component with different params.  It should now look
    // like two components, not combine with the others.
    ClosureColor *B = create_component (context, Color3(.4, .4, .4), 0.5f);
    // Create a closure with one component
    ClosureAdd *add = context->closure_add_allot (A, B);
    ASSERT (add);  // FIXME -- check this in a better way 
    /* We can't check this anymore
    c.flatten(&add->parent, shadingsys);

    ASSERT (c.ncomponents() == 2);
    ASSERT (c.weight(1) == Color3 (0.4, 0.4, 0.4));
    */
    //std::cout << "c = " << c << "\n";
    delete shadingsys;
}

#endif
