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

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of derivative related operations
///
/////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"
#include "dual.h"

#include "OpenImageIO/varyingref.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


template<class T, int whichd>
static DECLOP (specialized_Dxy)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));

    if (Src.is_varying() && Src.has_derivs()) {
        // Derivs always return varying data
        exec->adjust_varying (Result, true, Result.data() != Src.data());
        DASSERT (Result.is_varying());
        VaryingRef<T> result ((T *)Result.data(), Result.step());
        VaryingRef<Dual2<T> > src ((Dual2<T> *)Src.data(), Src.step());
        if (whichd == 0) {
            for (int i = beginpoint;  i < endpoint;  ++i)
                if (runflags[i])
                    result[i] = src[i].dx ();
        } else {
            for (int i = beginpoint;  i < endpoint;  ++i)
                if (runflags[i])
                    result[i] = src[i].dy ();
        }
        if (Result.has_derivs())
            exec->zero_derivs (Result);  // 2nd order derivs are always zero
    } else {
        // Src doesn't have derivs, so they're zero
        exec->adjust_varying (Result, false);
        exec->zero (Result);
    }
}



DECLOP (OP_Dx)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));

    OpImpl impl = NULL;
    if (Result.typespec().is_float() && Src.typespec().is_float())
        impl = specialized_Dxy<float, 0>;
    else if (Result.typespec().is_triple() && Src.typespec().is_triple())
        impl = specialized_Dxy<Vec3, 0>;

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
        return;
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Derivative type can't be handled");
    }
}



DECLOP (OP_Dy)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));

    OpImpl impl = NULL;
    if (Result.typespec().is_float() && Src.typespec().is_float())
        impl = specialized_Dxy<float, 1>;
    else if (Result.typespec().is_triple() && Src.typespec().is_triple())
        impl = specialized_Dxy<Vec3, 1>;

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
        return;
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Derivative type can't be handled");
    }
}



DECLOP (OP_calculatenormal)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &P (exec->sym (args[1]));
    DASSERT (Result.typespec().is_triple());
    DASSERT (P.typespec().is_triple());

    if (P.is_varying() && P.has_derivs()) {
        // output normal is always varying
        exec->adjust_varying (Result, true, Result.data() != P.data());
        DASSERT (Result.is_varying());
        VaryingRef<Vec3> result ((Vec3 *)Result.data(), Result.step());
        VaryingRef<Dual2<Vec3> > p ((Dual2<Vec3> *)P.data(), P.step());
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = p[i].dx().cross(p[i].dy());
        if (Result.has_derivs())
            exec->zero_derivs (Result);  // 2nd order derivs are always zero
    } else {
        // P doesn't have derivs, so we can't compute a good normal
        exec->adjust_varying (Result, false);
        exec->zero (Result);
    }
}


DECLOP (OP_area)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &P (exec->sym (args[1]));
    DASSERT (Result.typespec().is_float());
    DASSERT (P.typespec().is_triple());

    if (P.is_varying() && P.has_derivs()) {
        // differential area is always varying
        exec->adjust_varying (Result, true, Result.data() != P.data());
        DASSERT (Result.is_varying());
        VaryingRef<Float> result ((Float *)Result.data(), Result.step());
        VaryingRef<Dual2<Vec3> > p ((Dual2<Vec3> *)P.data(), P.step());
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = (p[i].dx().cross(p[i].dy())).length();
        if (Result.has_derivs())
            exec->zero_derivs (Result);  // 2nd order derivs are always zero
    } else {
        // P doesn't have derivs, so differential area is 0
        exec->adjust_varying (Result, false);
        exec->zero (Result);
    }
}



inline float
filter_width (float dx, float dy) {
    return sqrtf (dx * dx + dy * dy);
}

inline Vec3
filter_width (const Vec3 &dx, const Vec3 &dy) {
    return Vec3 (filter_width (dx.x, dy.x),
                 filter_width (dx.y, dy.y),
                 filter_width (dx.z, dy.z));
}

template <typename T>
DECLOP (filterwidth_guts)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));

    if (Src.is_varying() && Src.has_derivs()) {
        // differential area is always varying
        exec->adjust_varying (Result, true, Result.data() != Src.data());
        DASSERT (Result.is_varying());
        VaryingRef<T> result ((T *)Result.data(), Result.step());
        VaryingRef<Dual2<T> > src ((Dual2<T> *)Src.data(), Src.step());
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = filter_width(src[i].dx(), src[i].dy());
        if (Result.has_derivs())
            exec->zero_derivs (Result);  // 2nd order derivs are always zero
    } else {
        // P doesn't have derivs, so differential area is 0
        exec->adjust_varying (Result, false);
        exec->zero (Result);
    }
}


DECLOP (OP_filterwidth)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));

    OpImpl impl = NULL;
    if (Result.typespec().is_float() && Src.typespec().is_float())
        impl = filterwidth_guts<float>;
    else if (Result.typespec().is_triple() && Src.typespec().is_triple())
        impl = filterwidth_guts<Vec3>;

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
        return;
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Filterwidth type can't be handled");
    }
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
