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
/// Shader interpreter implementation of vector operations.
///
/////////////////////////////////////////////////////////////////////////


#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"


namespace OSL {
namespace pvt {


namespace {

/// Implementation of the constructor "triple (float, float, float)".
/// Since no coordinate system name is supplied, this will work with any
/// of the triple types.
DECLOP (triple_ctr)
{
    Symbol &Result (exec->sym (args[0]));
    Symbol &X (exec->sym (args[1]));
    Symbol &Y (exec->sym (args[2]));
    Symbol &Z (exec->sym (args[3]));

    // Adjust the result's uniform/varying status
    bool vary = (X.is_varying() | Y.is_varying() | Z.is_varying());
    exec->adjust_varying (Result, vary, false /* can't alias */);

    VaryingRef<Vec3> result ((Vec3 *)Result.data(), Result.step());
    VaryingRef<float> x ((float *)X.data(), X.step());
    VaryingRef<float> y ((float *)Y.data(), Y.step());
    VaryingRef<float> z ((float *)Z.data(), Z.step());

    if (result.is_uniform()) {
        // Everything is uniform
        *result = Vec3 (*x, *y, *z);
    } else if (! vary) {
        // Result is varying, but everything else is uniform
        Vec3 r (*x, *y, *z);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = Vec3 (x[i], y[i], z[i]);
    }
}



/// Implementation of the constructor "triple (string, float, float, float)".
/// Templated on the type of transformation needed (point, vector, normal).
template<int xformtype>
DECLOP (triple_ctr_transform)
{
    bool using_space = (nargs == 5);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Space (exec->sym (args[1]));
    Symbol &X (exec->sym (args[2]));
    Symbol &Y (exec->sym (args[3]));
    Symbol &Z (exec->sym (args[4]));

    // Adjust the result's uniform/varying status
    bool vary = (Space.is_varying() |
                 X.is_varying() | Y.is_varying() | Z.is_varying());
    exec->adjust_varying (Result, vary, false /* can't alias */);

    VaryingRef<Vec3> result ((Vec3 *)Result.data(), Result.step());
    VaryingRef<ustring> space ((ustring *)Space.data(), Space.step());
    VaryingRef<float> x ((float *)X.data(), X.step());
    VaryingRef<float> y ((float *)Y.data(), Y.step());
    VaryingRef<float> z ((float *)Z.data(), Z.step());

    if (result.is_uniform()) {
        // Everything is uniform
        *result = Vec3 (*x, *y, *z);
    } else if (! vary) {
        // Result is varying, but everything else is uniform
        Vec3 r (*x, *y, *z);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = Vec3 (x[i], y[i], z[i]);
    }
    ASSERT (0);  // FIXME -- not yet considering the space name!
}



/// Templated constructor for triples (point, vector, normal) with full
/// error checking and polymorphism resolution (based on the type of
/// transformation and whether a coordinate system name was supplied).
/// After doing the sanity checks, a specific implementation is chosen
/// and will be used directly for all subsequent calls to this op.
template<int xformtype>
DECLOP (triple_ctr_shadeop)
{
    ASSERT (nargs == 4 || nargs == 5);
    Symbol &Result (exec->sym (args[0]));
    bool using_space = (nargs == 5);
    Symbol &Space (exec->sym (args[1]));
    Symbol &X (exec->sym (args[1+using_space]));
    Symbol &Y (exec->sym (args[2+using_space]));
    Symbol &Z (exec->sym (args[3+using_space]));
    ASSERT (! Result.typespec().is_closure() && 
            ! X.typespec().is_closure() && ! Y.typespec().is_closure() &&
            ! Z.typespec().is_closure() && ! Space.typespec().is_closure());
    
    // We allow two flavors: point = point(float,float,float) and
    // point = point(string,float,float,float)
    if (Result.typespec().is_triple() && X.typespec().is_float() &&
          Y.typespec().is_float() && Z.typespec().is_float() &&
          (using_space == false || Space.typespec().is_string())) {
        OpImpl impl = NULL;
        if (using_space)
            impl = triple_ctr_transform<xformtype>;
        else
            impl = triple_ctr;
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        std::cerr << "Don't know how compute " << Result.typespec().string()
                  << " = " << exec->op().opname() << "("
                  << (using_space ? Space.typespec().string() : std::string())
                  << (using_space ? ", " : "")
                  << X.typespec().string() << ", "
                  << Y.typespec().string() << ", "
                  << Z.typespec().string() << ")\n";
        ASSERT (0 && "Function arg type can't be handled");
    }
}

};  // End anonymous namespace




DECLOP (OP_point)
{
    triple_ctr_shadeop<TypeDesc::POINT> (exec, nargs, args,
                                         runflags, beginpoint, endpoint);
}



DECLOP (OP_vector)
{
    triple_ctr_shadeop<TypeDesc::VECTOR> (exec, nargs, args,
                                          runflags, beginpoint, endpoint);
}



DECLOP (OP_normal)
{
    triple_ctr_shadeop<TypeDesc::NORMAL> (exec, nargs, args,
                                          runflags, beginpoint, endpoint);
}



}; // namespace pvt
}; // namespace OSL
