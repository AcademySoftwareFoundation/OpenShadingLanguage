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

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of assignment: '=' in all its
/// flavors.
///
/////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


// Heavy lifting of 'aref', this is a specialized version that knows
// the types of the arguments.
template <class T>
static DECLOP (specialized_aref)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));
    Symbol &Index (exec->sym (args[2]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, Src.is_varying() || Index.is_varying());

    // Loop over points, do the assignment.
    VaryingRef<T> result ((T *)Result.data(), Result.step());
    VaryingRef<T> src ((T *)Src.data(), Src.step());
    VaryingRef<int> index ((int *)Index.data(), Index.step());
    if (result.is_uniform()) {
        // Uniform case
        int ind = *index;
        *result = T ((&src[0])[ind]);
        if (Result.has_derivs ())
            exec->zero_derivs (Result);
    } else if (index.is_uniform()) {
        // Uniform index, potentially varying src array
        int ind = *index;
        SHADE_LOOP_BEGIN
            result[i] = T ((&src[i])[ind]);
        SHADE_LOOP_END
        if (Result.has_derivs ()) {
            if (Src.is_varying() && Src.has_derivs()) {
                int len = Src.typespec().arraylength();
                VaryingRef<Dual2<T> > result ((Dual2<T> *)Result.data(), Result.step());
                SHADE_LOOP_BEGIN
                    result[i].set_dx ((&src[i])[ind+len]);
                    result[i].set_dy ((&src[i])[ind+2*len]);
                SHADE_LOOP_END
            } else {
                exec->zero_derivs (Result);
            }
        }
    } else {
        // Fully varying case
        SHADE_LOOP_BEGIN
            result[i] = T ((&src[i])[index[i]]);
        SHADE_LOOP_END

        if (Result.has_derivs ()) {
            if (Src.is_varying() && Src.has_derivs()) {
                int len = Src.typespec().arraylength();
                VaryingRef<Dual2<T> > result ((Dual2<T> *)Result.data(), Result.step());
                SHADE_LOOP_BEGIN
                    result[i].set_dx ((&src[i])[index[i]+len]);
                    result[i].set_dy ((&src[i])[index[i]+2*len]);
                SHADE_LOOP_END
            } else {
                exec->zero_derivs (Result);
            }
        }
    }
}



DECLOP (OP_aref)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));
    Symbol &Index (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() && ! Src.typespec().is_closure());
    ASSERT (! Index.typespec().is_closure() && Index.typespec().is_int());
    ASSERT (Src.typespec().is_array() && ! Result.typespec().is_array() &&
            equivalent (Result.typespec(), Src.typespec().elementtype()));
    OpImpl impl = NULL;
    if (Result.typespec().is_float()) {
        impl = specialized_aref<float>;
    } else if (Result.typespec().is_int()) {
        impl = specialized_aref<int>;
    } else if (Result.typespec().is_triple()) {
        impl = specialized_aref<Vec3>;
    } else if (Result.typespec().is_matrix()) {
        impl = specialized_aref<Matrix44>;
    } else if (Result.typespec().is_string()) {
        impl = specialized_aref<ustring>;
    }
    if (impl) {
        impl (exec, nargs, args);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
        return;
    } else {
        exec->error ("Don't know how to assign %s = %s [%s]",
                     Result.typespec().c_str(), Src.typespec().c_str(),
                     Index.typespec().c_str());
        ASSERT (0 && "Array reference types can't be handled");
    }
}



// Heavy lifting of 'aassign', this is a specialized version that knows
// the types of the arguments.
template <class T>
static DECLOP (specialized_aassign)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &Index (exec->sym (args[1]));
    Symbol &Src (exec->sym (args[2]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, Src.is_varying() || Index.is_varying());

    // Loop over points, do the assignment.
    VaryingRef<T> result ((T *)Result.data(), Result.step());
    VaryingRef<int> index ((int *)Index.data(), Index.step());
    VaryingRef<T> src ((T *)Src.data(), Src.step());
    if (result.is_uniform()) {
        // Uniform case
        int ind = *index;
        (&result[0])[ind] = *src;
        if (Result.has_derivs ()) {
            // write 0's to this element only
            int len = Result.typespec().arraylength();
            memset(&((&result[0])[ind +     len]), 0, sizeof(T));
            memset(&((&result[0])[ind + 2 * len]), 0, sizeof(T));
        }
    } else if (index.is_uniform()) {
        // Uniform index, potentially varying src array
        int ind = *index;
        SHADE_LOOP_BEGIN
            (&result[i])[ind] = src[i];
        SHADE_LOOP_END

        if (Result.has_derivs()) {
            int len = Result.typespec().arraylength();
            if (Src.is_varying() && Src.has_derivs()) {
                VaryingRef<Dual2<T> > src ((Dual2<T> *)Src.data(), Src.step());
                SHADE_LOOP_BEGIN
                    (&result[i])[ind +     len] = src[i].dx();
                    (&result[i])[ind + 2 * len] = src[i].dy();
                SHADE_LOOP_END
            } else {
                SHADE_LOOP_BEGIN
                    memset(&((&result[i])[ind +     len]), 0, sizeof(T));
                    memset(&((&result[i])[ind + 2 * len]), 0, sizeof(T));
                SHADE_LOOP_END
            }
        }
    } else {
        // Fully varying case
        SHADE_LOOP_BEGIN
            (&result[i])[index[i]] = src[i];
        SHADE_LOOP_END

        if (Result.has_derivs()) {
            int len = Result.typespec().arraylength();
            if (Src.is_varying() && Src.has_derivs()) {
                VaryingRef<Dual2<T> > src ((Dual2<T> *)Src.data(), Src.step());
                SHADE_LOOP_BEGIN
                    (&result[i])[index[i] +     len] = src[i].dx();
                    (&result[i])[index[i] + 2 * len] = src[i].dy();
                SHADE_LOOP_END
            } else {
                SHADE_LOOP_BEGIN
                    memset(&((&result[i])[index[i] +     len]), 0, sizeof(T));
                    memset(&((&result[i])[index[i] + 2 * len]), 0, sizeof(T));
                SHADE_LOOP_END
            }
        }
    }
}



DECLOP (OP_aassign)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Index (exec->sym (args[1]));
    Symbol &Src (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() && ! Src.typespec().is_closure());
    ASSERT (! Index.typespec().is_closure() && Index.typespec().is_int());
    ASSERT (Result.typespec().is_array() && ! Src.typespec().is_array());
    OpImpl impl = NULL;
    TypeSpec Relement = Result.typespec().elementtype();
    if (Relement.is_float() && Src.typespec().is_float()) {
        impl = specialized_aassign<float>;
    } else if (Relement.is_int() && Src.typespec().is_int()) {
        impl = specialized_aassign<int>;
    } else if (Relement.is_triple() && Src.typespec().is_triple()) {
        impl = specialized_aassign<Vec3>;
    } else if (Relement.is_matrix() && Src.typespec().is_matrix()) {
        impl = specialized_aassign<Matrix44>;
    } else if (Relement.is_string() && Src.typespec().is_string()) {
        impl = specialized_aassign<ustring>;
    }
    if (impl) {
        impl (exec, nargs, args);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
        return;
    } else {
        exec->error ("Don't know how to assign %s [%s] = %s",
                     Relement.c_str(), Index.typespec().c_str(),
                     Src.typespec().c_str());
        ASSERT (0 && "Array assignment types can't be handled");
    }
}



DECLOP (OP_arraylength)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    DASSERT (Result.typespec().is_int() && A.typespec().is_array());
    
    // Result is always uniform!  (Though note that adjust_varying will
    // still make it varying if inside a conditional.)
    exec->adjust_varying (Result, false);

    VaryingRef<int> result ((int *)Result.data(), Result.step());
    int len = A.typespec().arraylength ();
    if (result.is_uniform()) {
        *result = len;
    } else {
        SHADE_LOOP_BEGIN
            result[i] = len;
        SHADE_LOOP_END
    }
    if (Result.has_derivs ())
        exec->zero_derivs (Result);   // arraylength not varying
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
