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

#include "oslops.h"
#include "oslexec_pvt.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


static Color3 one (1.0f, 1.0f, 1.0f);



DECLOP (OP_diffuse)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &N (exec->sym (args[1]));
    DASSERT (Result.typespec().is_closure());
    DASSERT (N.typespec().is_triple());

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, true /* closures always vary */);
    // N.B. Closures don't have derivs

    VaryingRef<ClosureColor *> result ((ClosureColor **)Result.data(), Result.step());
    VaryingRef<Vec3> n ((Vec3 *)N.data(), N.step());

    // Since diffuse takes no args, we can construct it just once.
    const ClosurePrimitive *prim = ClosurePrimitive::primitive (Strings::diffuse);
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            result[i]->set (prim);
            result[i]->set_parameter (0, 0, &(n[i]));
        }
    }
}



DECLOP (OP_transparent)
{
    DASSERT (nargs == 1);
    Symbol &Result (exec->sym (args[0]));
    DASSERT (Result.typespec().is_closure());

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, true /* closures always vary */);
    // N.B. Closures don't have derivs

    VaryingRef<ClosureColor *> result ((ClosureColor **)Result.data(), Result.step());

    // Since transparent takes no args, we can construct it just once.
    const ClosurePrimitive *prim = ClosurePrimitive::primitive (Strings::transparent);
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            result[i]->set (prim);
        }
    }
}



DECLOP (OP_phong)
{
    DASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &N (exec->sym (args[1]));
    Symbol &exponent (exec->sym (args[2]));
    DASSERT (Result.typespec().is_closure());
    DASSERT (N.typespec().is_triple());
    DASSERT (exponent.typespec().is_float());

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, true /* closures always vary */);
    // N.B. Closures don't have derivs

    VaryingRef<ClosureColor *> result ((ClosureColor **)Result.data(), Result.step());
    VaryingRef<Vec3> n ((Vec3 *)N.data(), N.step());
    VaryingRef<float> exp ((float *)exponent.data(), exponent.step());

    const ClosurePrimitive *prim = ClosurePrimitive::primitive (Strings::phong);
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            result[i]->set (prim);
            result[i]->set_parameter (0, 0, &(n[i]));
            result[i]->set_parameter (0, 1, &(exp[i]));
        }
    }
}



DECLOP (OP_ward)
{
    DASSERT (nargs == 5);
    Symbol &Result (exec->sym (args[0]));
    Symbol &N (exec->sym (args[1]));
    Symbol &T (exec->sym (args[2]));
    Symbol &Ax (exec->sym (args[3]));
    Symbol &Ay (exec->sym (args[4]));
    DASSERT (Result.typespec().is_closure());
    DASSERT (N.typespec().is_triple());
    DASSERT (T.typespec().is_triple());
    DASSERT (Ax.typespec().is_float());
    DASSERT (Ay.typespec().is_float());

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, true /* closures always vary */);
    // N.B. Closures don't have derivs

    VaryingRef<ClosureColor *> result ((ClosureColor **)Result.data(), Result.step());
    VaryingRef<Vec3> n ((Vec3 *)N.data(), N.step());
    VaryingRef<Vec3> t ((Vec3 *)T.data(), T.step());
    VaryingRef<float> ax ((float *)Ax.data(), Ax.step());
    VaryingRef<float> ay ((float *)Ay.data(), Ay.step());

    const ClosurePrimitive *prim = ClosurePrimitive::primitive (Strings::ward);
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            result[i]->set (prim);
            result[i]->set_parameter (0, 0, &(n[i]));
            result[i]->set_parameter (0, 1, &(t[i]));
            result[i]->set_parameter (0, 2, &(ax[i]));
            result[i]->set_parameter (0, 3, &(ay[i]));
        }
    }
}


DECLOP (OP_microfacet_ggx)
{
    DASSERT (nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &N (exec->sym (args[1]));
    Symbol &Ag (exec->sym (args[2]));
    Symbol &R0 (exec->sym (args[3]));
    DASSERT (Result.typespec().is_closure());
    DASSERT (N.typespec().is_triple());
    DASSERT (Ag.typespec().is_float());
    DASSERT (R0.typespec().is_float());

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, true /* closures always vary */);
    // N.B. Closures don't have derivs

    VaryingRef<ClosureColor *> result ((ClosureColor **)Result.data(), Result.step());
    VaryingRef<Vec3> n ((Vec3 *)N.data(), N.step());
    VaryingRef<float> ag ((float *)Ag.data(), Ag.step());
    VaryingRef<float> r0 ((float *)R0.data(), R0.step());

    const ClosurePrimitive *prim = ClosurePrimitive::primitive (Strings::microfacet_ggx);
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            result[i]->set (prim);
            result[i]->set_parameter (0, 0, &(n[i]));
            result[i]->set_parameter (0, 1, &(ag[i]));
            result[i]->set_parameter (0, 2, &(r0[i]));
        }
    }
}


DECLOP (OP_microfacet_beckmann)
{
    DASSERT (nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &N (exec->sym (args[1]));
    Symbol &Ab (exec->sym (args[2]));
    Symbol &R0 (exec->sym (args[3]));
    DASSERT (Result.typespec().is_closure());
    DASSERT (N.typespec().is_triple());
    DASSERT (Ab.typespec().is_float());
    DASSERT (R0.typespec().is_float());

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, true /* closures always vary */);
    // N.B. Closures don't have derivs

    VaryingRef<ClosureColor *> result ((ClosureColor **)Result.data(), Result.step());
    VaryingRef<Vec3> n ((Vec3 *)N.data(), N.step());
    VaryingRef<float> ab ((float *)Ab.data(), Ab.step());
    VaryingRef<float> r0 ((float *)R0.data(), R0.step());

    const ClosurePrimitive *prim = ClosurePrimitive::primitive (Strings::microfacet_beckmann);
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            result[i]->set (prim);
            result[i]->set_parameter (0, 0, &(n[i]));
            result[i]->set_parameter (0, 1, &(ab[i]));
            result[i]->set_parameter (0, 2, &(r0[i]));
        }
    }
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
