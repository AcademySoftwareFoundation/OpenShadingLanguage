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

#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"


namespace OSL {
namespace pvt {


// Proxy type that derives from Matrix44 but allows assignment of a float
// to mean f*Identity.
class MatrixProxy : public Matrix44 {
public:
    MatrixProxy (float a, float b, float c, float d,
                 float e, float f, float g, float h,
                 float i, float j, float k, float l,
                 float m, float n, float o, float p)
        : Matrix44 (a,b,c,d, e,f,g,h, i,j,k,l, m,n,o,p) { }

    MatrixProxy (float f) : Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f) { }

    const MatrixProxy& operator= (float f) {
        *this = MatrixProxy (f);
        return *this;
    }
};



// Heavy lifting of 'assign', this is a specialized version that knows
// the types of the arguments.
template <class RET, class SRC>
static DECLOP (specialized_assign)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, Src.is_varying(),
                          Result.data() == Src.data());

    // Loop over points, do the assignment.
    VaryingRef<RET> result ((RET *)Result.data(), Result.step());
    VaryingRef<SRC> src ((SRC *)Src.data(), Src.step());
    if (result.is_uniform()) {
        // Uniform case
        *result = RET (*src);
    } else {
        // Potentially varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = RET (src[i]);
    }
}



// Special version of assign for when the source and result are the same
// exact type, so we can just memcpy.
static DECLOP (assign_copy)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, Src.is_varying(),
                          Result.data() == Src.data());

    // Loop over points, do the assignment.
    size_t size = Result.size ();
    if (Result.is_uniform()) {
        // Uniform case
        memcpy (Result.data(), Src.data(), size);
    } else if (exec->all_points_on() && Src.is_varying()) {
        // Simple case where a single memcpy will do
        memcpy (Result.data(), Src.data(), size * exec->npoints());
    } else {
        // Potentially varying case
        VaryingRef<char> result ((char *)Result.data(), Result.step());
        VaryingRef<char> src ((char *)Src.data(), Src.step());
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                memcpy (&result[i], &src[i], size);
    }
}



DECLOP (OP_assign)
{
    ASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());   // Not yet
    ASSERT (! Src.typespec().is_closure() &&
            ! Src.typespec().is_structure() &&
            ! Src.typespec().is_array());   // Not yet
    OpImpl impl = NULL;
    if (Result.typespec().is_closure()) {
        // FIXME -- not handled yet
    } else if (Result.typespec().is_structure()) {
        // FIXME -- not handled yet
    } else if (Result.typespec().simpletype() == Src.typespec().simpletype()) {
        // Easy case -- the two types are exactly the same.  That's
        // always legal and has a very specialized implementation.  This
        // case handles f=f, p=p, v=v, n=n, c=c, s=s, m=m.
        impl = assign_copy;
    } else if (Result.typespec().is_float()) {
        if (Src.typespec().is_int())
            impl = specialized_assign<float,int>;
    } else if (Result.typespec().is_triple()) {
        if (Src.typespec().is_triple())
            impl = assign_copy;  // p=n, v=p, etc.
        else if (Src.typespec().is_float())
            impl = specialized_assign<Vec3,float>;
        else if (Src.typespec().is_int())
            impl = specialized_assign<Vec3,int>;
    } else if (Result.typespec().is_matrix()) {
        if (Src.typespec().is_float())
            impl = specialized_assign<MatrixProxy,float>;
        else if (Src.typespec().is_int())
            impl = specialized_assign<MatrixProxy,int>;
    }
    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
        return;
    } else {
        std::cerr << "Don't know how to assign " << Result.typespec().string()
                  << " = " << Src.typespec().string() << "\n";
        ASSERT (0 && "Assignment types can't be handled");
    }
}



}; // namespace pvt
}; // namespace OSL
