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

    if (Result.has_derivs ())
        exec->zero_derivs (Result);

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
        SHADE_LOOP_BEGIN
            result[i] = r;
        SHADE_LOOP_END
    } else {
        // Fully varying case
        SHADE_LOOP_BEGIN
            result[i] = Vec3 (x[i], y[i], z[i]);
        SHADE_LOOP_END

        if (Result.has_derivs()) {
            // Ugh, handle derivs piece-meal because we may get any subset
            // of x,y,z with or without derivs.
            VaryingRef<Dual2<Vec3> > result ((Dual2<Vec3> *)Result.data(), Result.step());
            for (int c = 0;  c < 3;  ++c) {
                // Call it 'x', but we're really looping over X, Y, Z
                Symbol &X (exec->sym (args[1+c]));  // get right symbol
                if (X.has_derivs ()) {
                    VaryingRef<Dual2<float> > x ((Dual2<float> *)X.data(), X.step());
                    SHADE_LOOP_BEGIN
                        result[i].dx()[c] = x[i].dx();
                        result[i].dy()[c] = x[i].dy();
                    SHADE_LOOP_END
                }
            }
        }
    }
}



namespace {  // anonymous


inline void
multVecMatrix (const Matrix44 &M, Dual2<Vec3> &in, Dual2<Vec3> &out)
{
    // Rearrange into a Vec3<Dual2<float> >
    Imath::Vec3<Dual2<float> > din, dout;
    for (int i = 0;  i < 3;  ++i)
        din[i].set (in.val()[i], in.dx()[i], in.dy()[i]);

    M.multVecMatrix (din, dout);

    // Rearrange back into Dual2<Vec3>
    out.set (Vec3 (dout[0].val(), dout[1].val(), dout[2].val()),
             Vec3 (dout[0].dx(),  dout[1].dx(),  dout[2].dx()),
             Vec3 (dout[0].dy(),  dout[1].dy(),  dout[2].dy()));
}



inline void
multDirMatrix (const Matrix44 &M, Dual2<Vec3> &in, Dual2<Vec3> &out)
{
    M.multDirMatrix (in.val(), out.val());
    M.multDirMatrix (in.dx(), out.dx());
    M.multDirMatrix (in.dy(), out.dy());
}



/// Implementation of transform (matrix, triple).
/// Templated on the type of transformation needed (point, vector, normal).
template<int xformtype>
DECLOP (triple_matrix_transform)
{
    DASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Matrix (exec->sym (args[1]));
    Symbol &V (exec->sym (args[2]));
    DASSERT (! Result.typespec().is_closure() &&
             Result.typespec().is_triple() &&
             ! V.typespec().is_closure() && V.typespec().is_triple() &&
             Matrix.typespec().is_matrix());

    // Adjust the result's uniform/varying status
    ShaderGlobals *globals = exec->context()->globals();
    bool vary = (Matrix.is_varying() | V.is_varying() |
                 globals->time.is_varying());
    exec->adjust_varying (Result, vary, Result.data() == V.data());

    VaryingRef<Vec3> result ((Vec3 *)Result.data(), Result.step());
    VaryingRef<Matrix44> matrix ((Matrix44 *)Matrix.data(), Matrix.step());
    VaryingRef<Vec3> v ((Vec3 *)V.data(), V.step());
    if (! vary) {
        // Computations are uniform.
        Vec3 r;
        if (xformtype == (int)TypeDesc::POINT)
            matrix[0].multVecMatrix (*v, r);
        else if (xformtype == (int)TypeDesc::VECTOR)
            matrix[0].multDirMatrix (*v, r);
        else
            matrix[0].inverse().transpose().multDirMatrix (*v, r);
        if (result.is_uniform())  // Don't loop if result is uniform
            result[0] = r;
        else {
            SHADE_LOOP_BEGIN
                result[i] = r;
            SHADE_LOOP_END
        }
        if (Result.has_derivs ())
            exec->zero_derivs (Result);
    } else if (Matrix.is_uniform()) {
        // V is varying, but matrix is not.  This means that we can
        // inverse-transpose it just once (for normals), and that we
        // need not consider its derivatives.
        VaryingRef<Dual2<Vec3> > dresult ((Dual2<Vec3> *)Result.data(), Result.step());
        VaryingRef<Dual2<Vec3> > dv ((Dual2<Vec3> *)V.data(), V.step());
        bool derivs = Result.has_derivs() && V.has_derivs();
        Matrix44 M (*matrix);
        if (xformtype == (int)TypeDesc::NORMAL)
            M = M.inverse().transpose();
        SHADE_LOOP_BEGIN
            if (derivs) {
                if (xformtype == (int)TypeDesc::POINT)
                    multVecMatrix (M, dv[i], dresult[i]);
                else
                    multDirMatrix (M, dv[i], dresult[i]);
            } else {
                // No derivs
                if (xformtype == (int)TypeDesc::POINT)
                    M.multVecMatrix (v[i], result[i]);
                else
                    M.multDirMatrix (v[i], result[i]);
            }
        SHADE_LOOP_END
        if (Result.has_derivs() && ! V.has_derivs())
            exec->zero_derivs (Result);
    } else {
        // Fully varying case
        VaryingRef<Dual2<Vec3> > dresult ((Dual2<Vec3> *)Result.data(), Result.step());
        VaryingRef<Dual2<Vec3> > dv ((Dual2<Vec3> *)V.data(), V.step());
        bool derivs = Result.has_derivs() && V.has_derivs();
        // FIXME? -- we are purposely not considering M having derivatives.
        // Give us the chills to consider the proper derivs of an 
        // inverse-transpose, and it seems like an exceedingly rare case
        // to construct a spatially-varying matrix and needs its derivs.
        // Hopefully nobody will ever complain.
        SHADE_LOOP_BEGIN
            if (derivs) {
                // V has derivs, but M does not
                Matrix44 M (matrix[i]);
                if (xformtype == (int)TypeDesc::NORMAL)
                    M = M.inverse().transpose();
                if (xformtype == (int)TypeDesc::POINT)
                    multVecMatrix (M, dv[i], dresult[i]);
                else
                    multDirMatrix (M, dv[i], dresult[i]);
            } else {
                // No derivs
                if (xformtype == (int)TypeDesc::POINT) {
                    matrix[i].multVecMatrix (v[i], result[i]);
                } else if (xformtype == (int)TypeDesc::VECTOR) {
                    matrix[i].multDirMatrix (v[i], result[i]);
                } else {
                    matrix[i].inverse().transpose().multDirMatrix (v[i], result[i]);
                }
            }
        SHADE_LOOP_END
        if (Result.has_derivs() && ! derivs)
            exec->zero_derivs (Result);
    }
}



/// Implementation of the constructor "triple (string, float, float, float)".
/// Templated on the type of transformation needed (point, vector, normal).
template<int xformtype>
DECLOP (triple_ctr_transform)
{
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

    if (Result.has_derivs ())
        exec->zero_derivs (Result);

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
        SHADE_LOOP_BEGIN
            result[i] = r;
        SHADE_LOOP_END
    } else {
        // Fully varying case
        ustring last_space;
        SHADE_LOOP_BEGIN
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
        SHADE_LOOP_END

        if (Result.has_derivs()) {
            // Ugh, handle derivs piece-meal because we may get any subset
            // of x,y,z with or without derivs.
            VaryingRef<Dual2<Vec3> > result ((Dual2<Vec3> *)Result.data(), Result.step());
            Dual2<float> zero (0.0f);
            VaryingRef<Dual2<float> > xdir, ydir, zdir;
            if (X.has_derivs ())
                xdir.init ((Dual2<float> *)X.data(), X.step());
            else
                xdir.init (&zero, 0);
            if (Y.has_derivs ())
                ydir.init ((Dual2<float> *)Y.data(), Y.step());
            else
                ydir.init (&zero, 0);
            if (Z.has_derivs ())
                zdir.init ((Dual2<float> *)Z.data(), Z.step());
            else
                zdir.init (&zero, 0);
            ustring last_space;
            SHADE_LOOP_BEGIN
                if (space[i] != last_space || globals->time.is_varying()) {
                    exec->get_matrix (M, space[i], i);
                    if (xformtype == (int)TypeDesc::NORMAL)
                        M = M.inverse().transpose();
                    last_space = space[i];
                }
                Vec3 dPdx (xdir[i].dx(), ydir[i].dx(), zdir[i].dx());
                Vec3 dPdy (xdir[i].dy(), ydir[i].dy(), zdir[i].dy());
                M.multDirMatrix (dPdx, result[i].dx());
                M.multDirMatrix (dPdy, result[i].dy());
            SHADE_LOOP_END
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
        impl (exec, nargs, args);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}



// Functor for component reference
class Compref {
public:
    Compref (ShadingExecution *exec) : m_exec (exec) { }
    void operator() (float &result, const Vec3 &v, int i) {
        if (i < 0 || i > 2) {
            const Symbol &V (m_exec->sym (m_exec->op().firstarg()+1));
            m_exec->error ("Index out of range: %s %s[%d]\n",
                           V.typespec().string().c_str(),
                           V.name().c_str(), i);
            i = clamp (i, 0, 2);
        }
        result = v[i];
    }
    void operator() (Dual2<float> &result, const Dual2<Vec3> &v, int i) {
        if (i < 0 || i > 2) {
            const Symbol &V (m_exec->sym (m_exec->op().firstarg()+1));
            m_exec->error ("Index out of range: %s %s[%d]\n",
                           V.typespec().string().c_str(),
                           V.name().c_str(), i);
            i = clamp (i, 0, 2);
        }
        result.set (v.val()[i], v.dx()[i], v.dy()[i]);
    }
private:
    ShadingExecution *m_exec;
};


class Dot {
public:
    Dot (ShadingExecution *) { }
    void operator() (float &result, const Vec3 &a, const Vec3 &b) { result = a.dot (b); }
    void operator() (Dual2<float> &result, const Dual2<Vec3> &a, const Dual2<Vec3> &b) {
        Dual2<float> ax = Dual2<float> (a.val().x, a.dx().x, a.dy().x);
        Dual2<float> ay = Dual2<float> (a.val().y, a.dx().y, a.dy().y);
        Dual2<float> az = Dual2<float> (a.val().z, a.dx().z, a.dy().z);
        Dual2<float> bx = Dual2<float> (b.val().x, b.dx().x, b.dy().x);
        Dual2<float> by = Dual2<float> (b.val().y, b.dx().y, b.dy().y);
        Dual2<float> bz = Dual2<float> (b.val().z, b.dx().z, b.dy().z);

        result = ax*bx + ay*by + az*bz;
    }
};


class Cross {
public:
    Cross (ShadingExecution *) { }
    void operator() (Vec3 &result, const Vec3 &a, const Vec3 &b) { result = a.cross (b); }
    void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &a, const Dual2<Vec3> &b) {
        Dual2<float> ax = Dual2<float> (a.val().x, a.dx().x, a.dy().x);
        Dual2<float> ay = Dual2<float> (a.val().y, a.dx().y, a.dy().y);
        Dual2<float> az = Dual2<float> (a.val().z, a.dx().z, a.dy().z);
        Dual2<float> bx = Dual2<float> (b.val().x, b.dx().x, b.dy().x);
        Dual2<float> by = Dual2<float> (b.val().y, b.dx().y, b.dy().y);
        Dual2<float> bz = Dual2<float> (b.val().z, b.dx().z, b.dy().z);

        Dual2<float> nx = ay*bz - az*by;
        Dual2<float> ny = az*bx - ax*bz;
        Dual2<float> nz = ax*by - ay*bx;

        result.set (Vec3(nx.val(), ny.val(), nz.val()),
                    Vec3(nx.dx(),  ny.dx(),  nz.dx()  ),
                    Vec3(nx.dy(),  ny.dy(),  nz.dy()  ));
    }
};


class Length {
public:
    Length (ShadingExecution *) { }
    void operator() (float &result, const Vec3 &a) { result = a.length(); }
    void operator() (Dual2<float> &result, const Dual2<Vec3> &a)
    {
        Dual2<float> ax = Dual2<float> (a.val().x, a.dx().x, a.dy().x);
        Dual2<float> ay = Dual2<float> (a.val().y, a.dx().y, a.dy().y);
        Dual2<float> az = Dual2<float> (a.val().z, a.dx().z, a.dy().z);
        result = sqrt(ax*ax + ay*ay + az*az);
    }
};


class Normalize {
public:
    Normalize (ShadingExecution *) { }
    void operator() (Vec3 &result, const Vec3 &a) { result = a.normalized(); }
    void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &a)
    {
        if (a.val().x == 0 && a.val().y == 0 && a.val().z == 0) {
            result.set (Vec3(0, 0, 0),
                        Vec3(0, 0, 0),
                        Vec3(0, 0, 0));
        } else {
            Dual2<float> ax = Dual2<float> (a.val().x, a.dx().x, a.dy().x);
            Dual2<float> ay = Dual2<float> (a.val().y, a.dx().y, a.dy().y);
            Dual2<float> az = Dual2<float> (a.val().z, a.dx().z, a.dy().z);
            Dual2<float> inv_length = 1.0f / sqrt(ax*ax + ay*ay + az*az);
            ax = ax*inv_length;
            ay = ay*inv_length;
            az = az*inv_length;
            result.set (Vec3(ax.val(), ay.val(), az.val()),
                        Vec3(ax.dx(),  ay.dx(),  az.dx() ),
                        Vec3(ax.dy(),  ay.dy(),  az.dy() ));
        }
    }
};


class Distance {
public:
    Distance (ShadingExecution *) { }
    void operator() (float &result, const Vec3 &a, const Vec3 &b) {
        float x = a[0] - b[0];
        float y = a[1] - b[1];
        float z = a[2] - b[2];
        result = sqrtf (x*x + y*y + z*z);
    }
    void operator() (Dual2<float> &result, const Dual2<Vec3> &a, const Dual2<Vec3> &b) {
        Dual2<float> ax = Dual2<float> (a.val().x, a.dx().x, a.dy().x);
        Dual2<float> ay = Dual2<float> (a.val().y, a.dx().y, a.dy().y);
        Dual2<float> az = Dual2<float> (a.val().z, a.dx().z, a.dy().z);
        Dual2<float> bx = Dual2<float> (b.val().x, b.dx().x, b.dy().x);
        Dual2<float> by = Dual2<float> (b.val().y, b.dx().y, b.dy().y);
        Dual2<float> bz = Dual2<float> (b.val().z, b.dx().z, b.dy().z);

        Dual2<float> dx = bx - ax;
        Dual2<float> dy = by - ay;
        Dual2<float> dz = bz - az;

        result = sqrt(dx*dx + dy*dy + dz*dz);
    }
};


};  // End anonymous namespace




DECLOP (OP_point)
{
    triple_ctr_shadeop<TypeDesc::POINT> (exec, nargs, args);
}



DECLOP (OP_vector)
{
    triple_ctr_shadeop<TypeDesc::VECTOR> (exec, nargs, args);
}



DECLOP (OP_normal)
{
    triple_ctr_shadeop<TypeDesc::NORMAL> (exec, nargs, args);
}



DECLOP (OP_transform)
{
    triple_matrix_transform<TypeDesc::POINT> (exec, nargs, args);
}



DECLOP (OP_transformv)
{
    triple_matrix_transform<TypeDesc::VECTOR> (exec, nargs, args);
}



DECLOP (OP_transformn)
{
    triple_matrix_transform<TypeDesc::NORMAL> (exec, nargs, args);
}



// result = vec[index]
DECLOP (OP_compref)
{
#ifdef DEBUG
    DASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &V (exec->sym (args[1]));
    Symbol &I (exec->sym (args[2]));
    DASSERT (! Result.typespec().is_closure() && 
             ! V.typespec().is_closure() && ! I.typespec().is_closure());
    DASSERT (Result.typespec().is_float() && V.typespec().is_triple() &&
             I.typespec().is_int());
#endif

    binary_op_unary_derivs<float,Vec3,int,Compref> (exec, nargs, args);
}



// result[index] = val
template<class SRC>
static DECLOP (specialized_compassign)
{
    Symbol &Result (exec->sym (args[0]));
    Symbol &Index (exec->sym (args[1]));
    Symbol &Val (exec->sym (args[2]));

    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, Result.is_varying() | Index.is_varying() | Val.is_varying());

    // Loop over points, do the operation
    if (Result.is_uniform()) {
        // Uniform case
        VaryingRef<Vec3> result ((Vec3 *)Result.data(), Result.step());
        VaryingRef<int> index ((int *)Index.data(), Index.step());
        VaryingRef<SRC> val ((SRC *)Val.data(), Val.step());
        int c = *index;
        if (c < 0 || c > 2) {
            exec->error ("Index out of range: %s %s[%d]\n",
                         Result.typespec().string().c_str(),
                         Result.name().c_str(), c);
            c = clamp (c, 0, 2);
        }
        (*result)[c] = (Float) *val;
        if (Result.has_derivs ())
            exec->zero_derivs (Result);
    } else {
        // Fully varying case -- but break out deriv support
        if (Result.has_derivs() && Val.has_derivs()) {
            VaryingRef<Dual2<Vec3> > result ((Dual2<Vec3> *)Result.data(), Result.step());
            VaryingRef<int> index ((int *)Index.data(), Index.step());
            VaryingRef<Dual2<SRC> > val ((Dual2<SRC> *)Val.data(), Val.step());
            SHADE_LOOP_BEGIN
                int c = index[i];
                if (c < 0 || c > 2) {
                    exec->error ("Index out of range: %s %s[%d]\n",
                                 Result.typespec().string().c_str(),
                                 Result.name().c_str(), c);
                    c = clamp (c, 0, 2);
                }
                Vec3 rval = result[i].val();
                Vec3 rdx  = result[i].dx();
                Vec3 rdy  = result[i].dy();
                rval[c] = (Float) val[i].val();
                rdx[c]  = (Float) val[i].dx();
                rdy[c]  = (Float) val[i].dy();
                result[i].set (rval, rdx, rdy);
            SHADE_LOOP_END
        } else {
            VaryingRef<Vec3> result ((Vec3 *)Result.data(), Result.step());
            VaryingRef<int> index ((int *)Index.data(), Index.step());
            VaryingRef<SRC> val ((SRC *)Val.data(), Val.step());
            SHADE_LOOP_BEGIN
                int c = index[i];
                if (c < 0 || c > 2) {
                    exec->error ("Index out of range: %s %s[%d]\n",
                                 Result.typespec().string().c_str(),
                                 Result.name().c_str(), c);
                    c = clamp (c, 0, 2);
                }
                result[i][c] = (Float) val[i];
            SHADE_LOOP_END
            if (Result.has_derivs ())
                exec->zero_derivs (Result);
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
        impl (exec, nargs, args);
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
    DASSERT (! exec->sym(args[0]).typespec().is_closure() &&
             ! exec->sym(args[1]).typespec().is_closure() && ! exec->sym(args[2]).typespec().is_closure());
    DASSERT (exec->sym(args[0]).typespec().is_float()  &&
             exec->sym(args[1]).typespec().is_triple() && exec->sym(args[2]).typespec().is_triple());

    binary_op<Float, Vec3, Vec3, Dot> (exec, nargs, args);
}



DECLOP (OP_cross)
{
    DASSERT (nargs == 3);
    DASSERT (! exec->sym(args[0]).typespec().is_closure() &&
             ! exec->sym(args[1]).typespec().is_closure() && ! exec->sym(args[2]).typespec().is_closure());
    DASSERT (exec->sym(args[0]).typespec().is_triple() &&
             exec->sym(args[1]).typespec().is_triple() && exec->sym(args[2]).typespec().is_triple());

    binary_op<Vec3, Vec3, Vec3, Cross> (exec, nargs, args);
}



DECLOP (OP_length)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    DASSERT (! Result.typespec().is_closure() && 
             ! A.typespec().is_closure());
    DASSERT (Result.typespec().is_float() && A.typespec().is_triple());

    unary_op_guts<Float,Vec3,Length> (Result, A, exec);
}



DECLOP (OP_normalize)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    DASSERT (! Result.typespec().is_closure() && 
             ! A.typespec().is_closure());
    DASSERT (Result.typespec().is_triple() && A.typespec().is_triple());

    unary_op_guts<Vec3,Vec3,Normalize> (Result, A, exec);
}



DECLOP (OP_distance)
{
    DASSERT (nargs == 3);
    DASSERT (! exec->sym(args[0]).typespec().is_closure() &&
             ! exec->sym(args[1]).typespec().is_closure() && ! exec->sym(args[2]).typespec().is_closure());
    DASSERT (exec->sym(args[0]).typespec().is_float()  &&
             exec->sym(args[1]).typespec().is_triple() && exec->sym(args[2]).typespec().is_triple());

    binary_op<Float,Vec3,Vec3,Distance> (exec, nargs, args);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
