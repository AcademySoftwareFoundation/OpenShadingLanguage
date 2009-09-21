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
/// Shader interpreter implementation of matrix operations.
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



/// matrix constructor.  Comes in several varieties:
///    matrix (float)
///    matrix (space, float)
///    matrix (...16 floats...)
///    matrix (space, ...16 floats...)
DECLOP (OP_matrix)
{
    Symbol &Result (exec->sym (args[0]));
    DASSERT (Result.typespec().is_matrix());
    Symbol &Space (exec->sym (args[1]));
    bool using_space = (nargs == 3 || nargs == 18);
    int nfloats = nargs - 1 - (int)using_space;
    DASSERT (nargs == 1 || nargs == 16);
    VaryingRef<Float> f[16];
    bool varying_args = false;
    for (int i = 0;  i < nfloats;  ++i) {
        Symbol &F (exec->sym (args[1+using_space+i]));
        varying_args |= F.is_varying();
        DASSERT (! F.typespec().is_closure() && F.typespec().is_float());
        f[i].init ((Float *) F.data(), F.step());
    }
    ShaderGlobals *globals = exec->context()->globals();
    if (using_space) {
        varying_args |= Space.is_varying();
        varying_args |= globals->time.is_varying();  // moving coord systems!
    }

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, varying_args, false /* can't alias */);

    // FIXME -- clear derivs for now, make it right later.
    if (Result.has_derivs ())
        exec->zero_derivs (Result);

    VaryingRef<Matrix44> result ((Matrix44 *)Result.data(), Result.step());
    VaryingRef<ustring> space ((ustring *)Space.data(), Space.step());
    if (! varying_args) {
        // Uniform case
        Matrix44 R;
        if (nfloats == 1) {
            Float a = *(f[0]);
            R = Matrix44 (a, 0, 0, 0, 0, a, 0, 0, 0, 0, a, 0, 0, 0, 0, a);
        } else {
            R = Matrix44 ( f[0][0],  f[1][0],  f[2][0],  f[3][0], 
                           f[4][0],  f[5][0],  f[6][0],  f[7][0], 
                           f[8][0],  f[9][0], f[10][0], f[11][0], 
                          f[12][0], f[13][0], f[14][0], f[15][0]);
        }
        if (using_space) {
            Matrix44 M;
            exec->get_matrix (M, space[0]);
            R = M * R;
        }
        if (result.is_uniform()) {
            // just one copy if result is uniform, too
            *result = R;
        } else {
            // Computation uniform, but copy to varying result variable
            for (int i = beginpoint;  i < endpoint;  ++i)
                if (runflags[i])
                    result[i] = R;
        }
    } else {
        // Fully varying case
        Matrix44 R, M;
        ustring last_space;
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i]) {
                if (nfloats == 1) {
                    Float a = f[0][i];
                    R = Matrix44 (a, 0, 0, 0, 0, a, 0, 0, 0, 0, a, 0, 0, 0, 0, a);
                } else {
                    R = Matrix44 ( f[0][i],  f[1][i],  f[2][i],  f[3][i], 
                                   f[4][i],  f[5][i],  f[6][i],  f[7][i], 
                                   f[8][i],  f[9][i], f[10][i], f[11][i], 
                                  f[12][i], f[13][i], f[14][i], f[15][i]);
                }
                if (using_space) {
                    if (space[i] != last_space || globals->time.is_varying()) {
                        exec->get_matrix (M, space[i], i);
                        last_space = space[i];
                    }
                    R = M * R;
                }
                result[i] = R;
            }
    }
}



// matrix[row][col] = val
template<class SRC>
static DECLOP (specialized_mxcompassign)
{
    Symbol &Result (exec->sym (args[0]));
    Symbol &Row (exec->sym (args[1]));
    Symbol &Col (exec->sym (args[2]));
    Symbol &Val (exec->sym (args[3]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, Result.is_varying() | Row.is_varying() |
                          Col.is_varying() | Val.is_varying());

    // FIXME -- clear derivs for now, make it right later.
    if (Result.has_derivs ())
        exec->zero_derivs (Result);

    // Loop over points, do the operation
    VaryingRef<Matrix44> result ((Matrix44 *)Result.data(), Result.step());
    VaryingRef<int> row ((int *)Row.data(), Row.step());
    VaryingRef<int> col ((int *)Col.data(), Col.step());
    VaryingRef<SRC> val ((SRC *)Val.data(), Val.step());
    if (result.is_uniform()) {
        // Uniform case
        int r = *row, c = *col;
        if (r < 0 || r > 3 || c < 0 || c > 3) {
            exec->error ("Index out of range: %s %s[%d][%d]\n",
                         Result.typespec().string().c_str(),
                         Result.name().c_str(), r, c);
            r = clamp (r, 0, 3);
            c = clamp (c, 0, 3);
        }
        (*result)[r][c] = (Float) *val;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i) {
            if (runflags[i]) {
                int r = row[i];
                int c = col[i];
                if (r < 0 || r > 3 || c < 0 || c > 3) {
                    exec->error ("Index out of range: %s %s[%d][%d]\n",
                                 Result.typespec().string().c_str(),
                                 Result.name().c_str(), r, c);
                    r = clamp (r, 0, 3);
                    c = clamp (c, 0, 3);
                }
                result[i][r][c] = (Float) val[i];
            }
        }
    }
}



// matrix[row][col] = val
DECLOP (OP_mxcompassign)
{
    ASSERT (nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Row (exec->sym (args[1]));
    Symbol &Col (exec->sym (args[2]));
    Symbol &Val (exec->sym (args[3]));
    ASSERT (! Result.typespec().is_closure() && ! Row.typespec().is_closure() &&
            ! Col.typespec().is_closure() && ! Val.typespec().is_closure());
    ASSERT (Result.typespec().is_matrix() && Row.typespec().is_int() &&
            Col.typespec().is_int());

    OpImpl impl = NULL;
    if (Val.typespec().is_float())
        impl = specialized_mxcompassign<Float>;
    else if (Val.typespec().is_int())
        impl = specialized_mxcompassign<int>;

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



// result = matrix[row][col]
DECLOP (OP_mxcompref)
{
    DASSERT (nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &M (exec->sym (args[1]));
    Symbol &Row (exec->sym (args[2]));
    Symbol &Col (exec->sym (args[3]));
    DASSERT (! Result.typespec().is_closure() && ! Row.typespec().is_closure() &&
            ! Col.typespec().is_closure() && ! M.typespec().is_closure());
    DASSERT (Result.typespec().is_float() && M.typespec().is_matrix() &&
             Row.typespec().is_int() && Col.typespec().is_int());

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, M.is_varying() | Row.is_varying() |
                          Col.is_varying(), false /* can't alias */);

    // FIXME -- clear derivs for now, make it right later.
    if (Result.has_derivs ())
        exec->zero_derivs (Result);

    // Loop over points, do the operation
    VaryingRef<Float> result ((Float *)Result.data(), Result.step());
    VaryingRef<Matrix44> m ((Matrix44 *)M.data(), M.step());
    VaryingRef<int> row ((int *)Row.data(), Row.step());
    VaryingRef<int> col ((int *)Col.data(), Col.step());
    if (result.is_uniform()) {
        // Uniform case
        int r = *row, c = *col;
        if (r < 0 || r > 3 || c < 0 || c > 3) {
            exec->error ("Index out of range: %s %s[%d][%d]\n",
                         Result.typespec().string().c_str(),
                         Result.name().c_str(), r, c);
            r = clamp (r, 0, 3);
            c = clamp (c, 0, 3);
        }
        (*result) = (*m)[r][c];
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i) {
            if (runflags[i]) {
                int r = row[i];
                int c = col[i];
                if (r < 0 || r > 3 || c < 0 || c > 3) {
                    exec->error ("Index out of range: %s %s[%d][%d]\n",
                                 Result.typespec().string().c_str(),
                                 Result.name().c_str(), r, c);
                    r = clamp (r, 0, 3);
                    c = clamp (c, 0, 3);
                }
                result[i] = (m[i])[r][c];
            }
        }
    }
}



namespace { // anonymous

// N.B. Determinant implementation graciously contributed by Marcos Fajardo.

// Calculate the determinant of a 2x2 matrix.
template <typename F>
inline F det2x2(F a, F b, F c, F d)
{
    return a * d - b * c;
}

// calculate the determinant of a 3x3 matrix in the form:
//     | a1,  b1,  c1 |
//     | a2,  b2,  c2 |
//     | a3,  b3,  c3 |
template <typename F>
inline F det3x3(F a1, F a2, F a3, F b1, F b2, F b3, F c1, F c2, F c3)
{
    return a1 * det2x2( b2, b3, c2, c3 )
         - b1 * det2x2( a2, a3, c2, c3 )
         + c1 * det2x2( a2, a3, b2, b3 );
}

// calculate the determinant of a 4x4 matrix.
template <typename F>
inline F det4x4(const Imath::Matrix44<F> &m)
{
    // assign to individual variable names to aid selecting correct elements
    F a1 = m[0][0], b1 = m[0][1], c1 = m[0][2], d1 = m[0][3];
    F a2 = m[1][0], b2 = m[1][1], c2 = m[1][2], d2 = m[1][3];
    F a3 = m[2][0], b3 = m[2][1], c3 = m[2][2], d3 = m[2][3];
    F a4 = m[3][0], b4 = m[3][1], c4 = m[3][2], d4 = m[3][3];
    return a1 * det3x3( b2, b3, b4, c2, c3, c4, d2, d3, d4)
         - b1 * det3x3( a2, a3, a4, c2, c3, c4, d2, d3, d4)
         + c1 * det3x3( a2, a3, a4, b2, b3, b4, d2, d3, d4)
         - d1 * det3x3( a2, a3, a4, b2, b3, b4, c2, c3, c4);
}


class Determinant {
public:
    Determinant (ShadingExecution *) { }
    inline float operator() (const Matrix44 &m) {
        return det4x4 (m);
    }
};



class Transpose {
public:
    Transpose (ShadingExecution *) { }
    inline Matrix44 operator() (const Matrix44 &m) {
        return m.transposed();
    }
};


};  // anonymous namespace


DECLOP (OP_determinant)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    DASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure());
    DASSERT (Result.typespec().is_float() && A.typespec().is_matrix());

    unary_op_guts<Float, Matrix44, Determinant> (Result, A, exec, runflags,
                                                 beginpoint, endpoint);
}



DECLOP (OP_transpose)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    DASSERT (! Result.typespec().is_closure() && ! A.typespec().is_closure());
    DASSERT (Result.typespec().is_matrix() && A.typespec().is_matrix());

    unary_op_guts<Matrix44, Matrix44, Transpose> (Result, A, exec, runflags,
                                                  beginpoint, endpoint);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
