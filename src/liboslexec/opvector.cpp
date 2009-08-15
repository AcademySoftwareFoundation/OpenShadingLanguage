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
#include "OpenImageIO/fmath.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


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



namespace {  // anonymous

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
    ShaderGlobals *globals = exec->context()->globals();
    bool vary = (Space.is_varying() |
                 X.is_varying() | Y.is_varying() | Z.is_varying() |
                 globals->time.is_varying());
    exec->adjust_varying (Result, vary, false /* can't alias */);

    VaryingRef<Vec3> result ((Vec3 *)Result.data(), Result.step());
    VaryingRef<ustring> space ((ustring *)Space.data(), Space.step());
    VaryingRef<float> x ((float *)X.data(), X.step());
    VaryingRef<float> y ((float *)Y.data(), Y.step());
    VaryingRef<float> z ((float *)Z.data(), Z.step());

    Matrix44 M;
    if (result.is_uniform()) {
        // Everything is uniform
        exec->get_matrix (M, *space);
        if (xformtype == (int)TypeDesc::NORMAL)
            M = M.inverse().transpose();
        *result = Vec3 (*x, *y, *z);
        if (xformtype == (int)TypeDesc::POINT)
            M.multVecMatrix (*result, *result);
        else
            M.multDirMatrix (*result, *result);
    } else if (! vary) {
        // Result is varying, but everything else is uniform
        exec->get_matrix (M, *space);
        if (xformtype == (int)TypeDesc::NORMAL)
            M = M.inverse().transpose();
        Vec3 r (*x, *y, *z);
        if (xformtype == (int)TypeDesc::POINT)
            M.multVecMatrix (r, r);
        else
            M.multDirMatrix (r, r);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        ustring last_space;
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i]) {
                if (space[i] != last_space || globals->time.is_varying()) {
                    exec->get_matrix (M, space[i], i);
                    if (xformtype == (int)TypeDesc::NORMAL)
                        M = M.inverse().transpose();
                    last_space = space[i];
                }
                result[i] = Vec3 (x[i], y[i], z[i]);
                if (xformtype == (int)TypeDesc::POINT)
                    M.multVecMatrix (result[i], result[i]);
                else
                    M.multDirMatrix (result[i], result[i]);
            }
    }
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



// Functor for component reference
class Compref {
public:
    Compref (ShadingExecution *exec) : m_exec (exec) { }
    float operator() (const Vec3 &v, int i) {
        if (i < 0 || i > 2) {
            const Symbol &V (m_exec->sym (m_exec->op().firstarg()+1));
            m_exec->error ("Index out of range: %s %s[%d]\n",
                           V.typespec().string().c_str(),
                           V.name().c_str(), i);
            i = clamp (i, 0, 2);
        }
        return v[i];
    }
private:
    ShadingExecution *m_exec;
};


class Dot {
public:
    Dot (ShadingExecution *) { }
    float operator() (const Vec3 &a, const Vec3 &b) { return a.dot (b); }
};


class Cross {
public:
    Cross (ShadingExecution *) { }
    Vec3 operator() (const Vec3 &a, const Vec3 &b) { return a.cross (b); }
};


class Length {
public:
    Length (ShadingExecution *) { }
    float operator() (const Vec3 &a) { return a.length(); }
};


class Normalize {
public:
    Normalize (ShadingExecution *) { }
    Vec3 operator() (const Vec3 &a) { return a.normalized(); }
};


class Distance {
public:
    Distance (ShadingExecution *) { }
    float operator() (const Vec3 &a, const Vec3 &b) {
        float x = a[0] - b[0];
        float y = a[1] - b[1];
        float z = a[2] - b[2];
        return sqrtf (x*x + y*y + z*z);
    }
};


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



// result = vec[index]
DECLOP (OP_compref)
{
    DASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &V (exec->sym (args[1]));
    Symbol &I (exec->sym (args[2]));
    DASSERT (! Result.typespec().is_closure() && 
             ! V.typespec().is_closure() && ! I.typespec().is_closure());
    DASSERT (Result.typespec().is_float() && V.typespec().is_triple() &&
             I.typespec().is_int());

    binary_op_guts<Float,Vec3,int,Compref> (Result, V, I, exec,
                                            runflags, beginpoint, endpoint);
}



// result[index] = val
template<class SRC>
static DECLOP (specialized_compassign)
{
    Symbol &Result (exec->sym (args[0]));
    Symbol &Index (exec->sym (args[1]));
    Symbol &Val (exec->sym (args[2]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, Result.is_varying() | Index.is_varying() | Val.is_varying(), false /* can't alias */);

    // Loop over points, do the operation
    VaryingRef<Vec3> result ((Vec3 *)Result.data(), Result.step());
    VaryingRef<int> index ((int *)Index.data(), Index.step());
    VaryingRef<SRC> val ((SRC *)Val.data(), Val.step());
    if (result.is_uniform()) {
        // Uniform case
        (*result)[*index] = (Float) *val;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i]) {
                result[i][index[i]] = (Float) val[i];
            }
    }
}



// result[index] = val
DECLOP (OP_compassign)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Index (exec->sym (args[1]));
    Symbol &Val (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() && 
            ! Index.typespec().is_closure() && ! Val.typespec().is_closure());
    ASSERT (Result.typespec().is_triple() && Index.typespec().is_int());

    OpImpl impl = NULL;
    if (Val.typespec().is_float())
        impl = specialized_compassign<Float>;
    else if (Val.typespec().is_int())
        impl = specialized_compassign<int>;

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
        return;
    } else {
        ASSERT (0 && "Component assignment types can't be handled");
    }
}



DECLOP (OP_dot)
{
    DASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    DASSERT (! Result.typespec().is_closure() && 
             ! A.typespec().is_closure() && ! B.typespec().is_closure());
    DASSERT (Result.typespec().is_float() && A.typespec().is_triple() &&
             B.typespec().is_triple());

    binary_op_guts<Float,Vec3,Vec3,Dot> (Result, A, B, exec,
                                         runflags, beginpoint, endpoint);
}



DECLOP (OP_cross)
{
    DASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    DASSERT (! Result.typespec().is_closure() && 
             ! A.typespec().is_closure() && ! B.typespec().is_closure());
    DASSERT (Result.typespec().is_triple() && A.typespec().is_triple() &&
             B.typespec().is_triple());

    binary_op_guts<Vec3,Vec3,Vec3,Cross> (Result, A, B, exec,
                                          runflags, beginpoint, endpoint);
}



DECLOP (OP_length)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    DASSERT (! Result.typespec().is_closure() && 
             ! A.typespec().is_closure());
    DASSERT (Result.typespec().is_float() && A.typespec().is_triple());

    unary_op_guts<Float,Vec3,Length> (Result, A, exec,
                                      runflags, beginpoint, endpoint);
}



DECLOP (OP_normalize)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    DASSERT (! Result.typespec().is_closure() && 
             ! A.typespec().is_closure());
    DASSERT (Result.typespec().is_triple() && A.typespec().is_triple());

    unary_op_guts<Vec3,Vec3,Normalize> (Result, A, exec,
                                        runflags, beginpoint, endpoint);
}



DECLOP (OP_distance)
{
    DASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    DASSERT (! Result.typespec().is_closure() && 
             ! A.typespec().is_closure() && ! B.typespec().is_closure());
    DASSERT (Result.typespec().is_float() && A.typespec().is_triple() &&
             B.typespec().is_triple());

    binary_op_guts<Float,Vec3,Vec3,Distance> (Result, A, B, exec,
                                              runflags, beginpoint, endpoint);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
