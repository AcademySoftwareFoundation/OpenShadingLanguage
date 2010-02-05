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

#ifndef OSLOPS_H
#define OSLOPS_H

#include "OpenImageIO/typedesc.h"

#include "oslexec.h"
#include "osl_pvt.h"
#include "oslexec_pvt.h"
#include "dual.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {


/// Macro that defines the arguments to shading opcode implementations
///
#define OPARGSDECL     ShadingExecution *exec, int nargs, const int *args, \
                       Runflag *runflags, int beginpoint, int endpoint

/// Macro that defines the full declaration of a shading opcode
/// implementation
#define DECLOP(name)   void name (OPARGSDECL)


// Declarations of all our shader opcodes follow:

//DECLOP (OP_aastep);
DECLOP (OP_aassign);
DECLOP (OP_acos);
DECLOP (OP_add);
//DECLOP (OP_ambient);
DECLOP (OP_and);
DECLOP (OP_ashikhmin_velvet);
DECLOP (OP_area);
DECLOP (OP_aref);
DECLOP (OP_arraylength);
DECLOP (OP_asin);
DECLOP (OP_atan);
DECLOP (OP_atan2);
DECLOP (OP_assign);
DECLOP (OP_background);
DECLOP (OP_bitand);
DECLOP (OP_bitor);
DECLOP (OP_bssrdf_cubic);
//DECLOP (OP_bump);
DECLOP (OP_calculatenormal);
DECLOP (OP_ceil);
DECLOP (OP_cellnoise);
DECLOP (OP_clamp);
DECLOP (OP_cloth);
DECLOP (OP_color);
DECLOP (OP_compassign);
DECLOP (OP_compl);
DECLOP (OP_compref);
DECLOP (OP_concat);
//DECLOP (OP_cooktorrance);
DECLOP (OP_cos);
DECLOP (OP_cosh);
DECLOP (OP_cross);
//DECLOP (OP_decr);
DECLOP (OP_degrees);
//DECLOP (OP_deltau);
//DECLOP (OP_deltav);
DECLOP (OP_determinant);
DECLOP (OP_dielectric);
DECLOP (OP_diffuse);
//DECLOP (OP_displace);
DECLOP (OP_distance);
DECLOP (OP_div);
DECLOP (OP_dot);
DECLOP (OP_dowhile);
DECLOP (OP_Dx);
DECLOP (OP_Dy);
DECLOP (OP_emission);
//DECLOP (OP_environment);
DECLOP (OP_end);
DECLOP (OP_endswith);
DECLOP (OP_erf);
DECLOP (OP_erfc);
DECLOP (OP_error);
//DECLOP (OP_exit);
DECLOP (OP_exp);
DECLOP (OP_exp2);
DECLOP (OP_expm1);
DECLOP (OP_eq);
DECLOP (OP_fabs);
//DECLOP (OP_faceforward);
DECLOP (OP_filterwidth);
DECLOP (OP_floor);
//DECLOP (OP_fmod);  // alias for OP_mod
DECLOP (OP_for);
DECLOP (OP_format);
//DECLOP (OP_fprintf);
DECLOP (OP_fresnel);
DECLOP (OP_ge);
DECLOP (OP_getattribute);
DECLOP (OP_getmessage);
DECLOP (OP_gettextureinfo);
DECLOP (OP_gt);
DECLOP (OP_hair_diffuse);
DECLOP (OP_hair_specular);
//DECLOP (OP_hash);
DECLOP (OP_hypot);
DECLOP (OP_if);
//DECLOP (OP_incr);
//DECLOP (OP_inversespline);
DECLOP (OP_inversesqrt);
DECLOP (OP_phong_ramp);
DECLOP (OP_isnan);
DECLOP (OP_isinf);
DECLOP (OP_iscameraray);
//DECLOP (OP_isindirectray);
DECLOP (OP_isfinite);
DECLOP (OP_isshadowray);
DECLOP (OP_le);
DECLOP (OP_length);
DECLOP (OP_log);
DECLOP (OP_log2);
DECLOP (OP_log10);
DECLOP (OP_logb);
DECLOP (OP_lt);
DECLOP (OP_luminance);
DECLOP (OP_matrix);
DECLOP (OP_max);
DECLOP (OP_microfacet_beckmann);
DECLOP (OP_microfacet_beckmann_refraction);
DECLOP (OP_microfacet_ggx);
DECLOP (OP_microfacet_ggx_refraction);
DECLOP (OP_min);
DECLOP (OP_mix);
DECLOP (OP_mxcompassign);
DECLOP (OP_mxcompref);
DECLOP (OP_mod);
DECLOP (OP_mul);
DECLOP (OP_neq);
DECLOP (OP_neg);
DECLOP (OP_noise);
DECLOP (OP_nop);
DECLOP (OP_normal);
DECLOP (OP_normalize);
DECLOP (OP_or);
//DECLOP (OP_orennayar);
DECLOP (OP_phong);
DECLOP (OP_pnoise);
DECLOP (OP_point);
DECLOP (OP_pow);
DECLOP (OP_printf);
DECLOP (OP_psnoise);
DECLOP (OP_radians);
//DECLOP (OP_random);
//DECLOP (OP_raylevel);
DECLOP (OP_reflect);
DECLOP (OP_reflection);
DECLOP (OP_refract);
DECLOP (OP_refraction);
DECLOP (OP_regex_match);
DECLOP (OP_regex_search);
//DECLOP (OP_rotate);
DECLOP (OP_round);
DECLOP (OP_setmessage);
//DECLOP (OP_shadow);
DECLOP (OP_shl);
DECLOP (OP_shr);
DECLOP (OP_sign);
DECLOP (OP_sin);
DECLOP (OP_sinh);
DECLOP (OP_smoothstep);
DECLOP (OP_snoise);
//DECLOP (OP_spline);
DECLOP (OP_sqrt);
DECLOP (OP_startswith);
DECLOP (OP_step);
DECLOP (OP_strlen);
DECLOP (OP_sub);
DECLOP (OP_substr);
//DECLOP (OP_subsurface);
DECLOP (OP_surfacearea);
DECLOP (OP_tan);
DECLOP (OP_tanh);
DECLOP (OP_texture);
DECLOP (OP_transform);
//DECLOP (OP_transformc);
DECLOP (OP_transformn);
//DECLOP (OP_transformu);
DECLOP (OP_transformv);
DECLOP (OP_transparent);
DECLOP (OP_transpose);
DECLOP (OP_translucent);
DECLOP (OP_trunc);
DECLOP (OP_useparam);
DECLOP (OP_vector);
DECLOP (OP_ward);
DECLOP (OP_warning);
DECLOP (OP_westin_backscatter);
DECLOP (OP_westin_sheen);
DECLOP (OP_xor);

DECLOP (OP_missing);


// Heavy lifting of the math and other ternary ops, this is a templated
// version that knows the types of the arguments and the operation to
// perform (given by a functor).
template <class RET, class ATYPE, class BTYPE, class CTYPE, class DTYPE, class FUNCTION>
inline void
quaternary_op_guts (Symbol &Result, Symbol &A, Symbol &B, Symbol &C, Symbol &D,
                    ShadingExecution *exec, 
                    Runflag *runflags, int beginpoint, int endpoint,
                    bool zero_derivs=true)
{
    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, A.is_varying() | B.is_varying() | C.is_varying() | D.is_varying(),
                          A.data() == Result.data() || B.data() == Result.data() ||
                          C.data() == Result.data() || D.data() == Result.data());

    // Loop over points, do the operation
    VaryingRef<RET> result ((RET *)Result.data(), Result.step());
    VaryingRef<ATYPE> a ((ATYPE *)A.data(), A.step());
    VaryingRef<BTYPE> b ((BTYPE *)B.data(), B.step());
    VaryingRef<CTYPE> c ((CTYPE *)C.data(), C.step());
    VaryingRef<DTYPE> d ((DTYPE *)D.data(), D.step());
    FUNCTION function (exec);
    if (result.is_uniform()) {
        // Uniform case
        function (*result, *a, *b, *c, *d);
    } else if (A.is_uniform() && B.is_uniform() && C.is_uniform()) {
        // Operands are uniform but we're assigning to a varying (it can
        // happen if we're in a conditional).  Take a shortcut by doing
        // the operation only once.
        RET r;
        function (r, *a, *b, *c, *d);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                function (result[i], a[i], b[i], c[i], d[i]);
    }

    if (zero_derivs && Result.has_derivs ())
        exec->zero_derivs (Result);
}



// this is a quaternary function where only the "A" and "B" arguments contribute
// to the result's derivatives
template <typename RET, typename ATYPE, typename BTYPE,
          typename CTYPE, typename DTYPE, typename FUNCTION>
DECLOP (quaternary_op_binary_derivs)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    Symbol &C (exec->sym (args[3]));
    Symbol &D (exec->sym (args[4]));

    if (Result.has_derivs()) {
        if (A.has_derivs()) {
            if (B.has_derivs())
                quaternary_op_guts<Dual2<RET>,Dual2<ATYPE>,Dual2<BTYPE>,CTYPE,DTYPE,FUNCTION> (Result, A, B, C, D, exec,
                        runflags, beginpoint, endpoint, false);
            else
                quaternary_op_guts<Dual2<RET>,Dual2<ATYPE>,BTYPE,CTYPE,DTYPE,FUNCTION> (Result, A, B, C, D, exec,
                        runflags, beginpoint, endpoint, false);
        } else if (B.has_derivs()) {
            quaternary_op_guts<Dual2<RET>,ATYPE,Dual2<BTYPE>,CTYPE,DTYPE,FUNCTION> (Result, A, B, C, D, exec,
                    runflags, beginpoint, endpoint, false);
        } else {
            quaternary_op_guts<RET,ATYPE,BTYPE,CTYPE,DTYPE,FUNCTION> (Result, A, B, C, D, exec,
                    runflags, beginpoint, endpoint,true);
        }
    } else {
        quaternary_op_guts<RET,ATYPE,BTYPE,CTYPE,DTYPE,FUNCTION> (Result, A, B, C, D, exec,
                runflags, beginpoint, endpoint, false);
    }
}



// Heavy lifting of the math and other ternary ops, this is a templated
// version that knows the types of the arguments and the operation to
// perform (given by a functor).
template <class RET, class ATYPE, class BTYPE, class CTYPE, class FUNCTION>
inline void
ternary_op_guts (Symbol &Result, Symbol &A, Symbol &B, Symbol &C,
                ShadingExecution *exec, 
                Runflag *runflags, int beginpoint, int endpoint,
                bool zero_derivs=true)
{
    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, A.is_varying() | B.is_varying() | C.is_varying(),
                          A.data() == Result.data() || B.data() == Result.data() || C.data() == Result.data());

    // Loop over points, do the operation
    VaryingRef<RET> result ((RET *)Result.data(), Result.step());
    VaryingRef<ATYPE> a ((ATYPE *)A.data(), A.step());
    VaryingRef<BTYPE> b ((BTYPE *)B.data(), B.step());
    VaryingRef<CTYPE> c ((CTYPE *)C.data(), C.step());
    FUNCTION function (exec);
    if (result.is_uniform()) {
        // Uniform case
        function (*result, *a, *b, *c);
    } else if (A.is_uniform() && B.is_uniform() && C.is_uniform()) {
        // Operands are uniform but we're assigning to a varying (it can
        // happen if we're in a conditional).  Take a shortcut by doing
        // the operation only once.
        RET r;
        function (r, *a, *b, *c);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                function (result[i], a[i], b[i], c[i]);
    }
    if (zero_derivs && Result.has_derivs ())
        exec->zero_derivs (Result);
}

// Wrapper around ternary_op_guts that does has he call signature of an
// ordinary shadeop.
template <class RET, class ATYPE, class BTYPE, class CTYPE, class FUNCTION>
DECLOP (ternary_op_noderivs)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    Symbol &C (exec->sym (args[3]));

    ternary_op_guts<RET,ATYPE,BTYPE,CTYPE,FUNCTION> (Result, A, B, C, exec,
                                              runflags, beginpoint, endpoint);
}

// Wrapper around ternary_op_guts that does has he call signature of an
// ordinary shadeop, with support for derivatives
template <class RET, class ATYPE, class BTYPE, class CTYPE, class FUNCTION>
DECLOP (ternary_op)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    Symbol &C (exec->sym (args[3]));

    if (Result.has_derivs()) {
        if (A.has_derivs()) {
           if (B.has_derivs()) {
               if (C.has_derivs()) {
                   ternary_op_guts<Dual2<RET>, Dual2<ATYPE>, Dual2<BTYPE>, Dual2<CTYPE>, FUNCTION>
                       (Result, A, B, C, exec, runflags, beginpoint, endpoint, false);
               } else {
                   ternary_op_guts<Dual2<RET>, Dual2<ATYPE>, Dual2<BTYPE>, CTYPE, FUNCTION>
                       (Result, A, B, C, exec, runflags, beginpoint, endpoint, false);
               }
           }
           else {
               if (C.has_derivs()) {
                   ternary_op_guts<Dual2<RET>, Dual2<ATYPE>, BTYPE, Dual2<CTYPE>, FUNCTION>
                       (Result, A, B, C, exec, runflags, beginpoint, endpoint, false);
               } else {
                   ternary_op_guts<Dual2<RET>, Dual2<ATYPE>, BTYPE, CTYPE, FUNCTION>
                       (Result, A, B, C, exec, runflags, beginpoint, endpoint, false);
               }
           }
        }
        else {
           if (B.has_derivs()) {
               if (C.has_derivs()) {
                   ternary_op_guts<Dual2<RET>, ATYPE, Dual2<BTYPE>, Dual2<CTYPE>, FUNCTION>
                       (Result, A, B, C, exec, runflags, beginpoint, endpoint, false);
               } else {
                   ternary_op_guts<Dual2<RET>, ATYPE, Dual2<BTYPE>, CTYPE, FUNCTION>
                       (Result, A, B, C, exec, runflags, beginpoint, endpoint, false);
               }
           } else {
               if (C.has_derivs()) {
                   ternary_op_guts<Dual2<RET>, ATYPE, BTYPE, Dual2<CTYPE>, FUNCTION>
                       (Result, A, B, C, exec, runflags, beginpoint, endpoint, false);
               } else {
                   ternary_op_guts<Dual2<RET>, ATYPE, BTYPE, CTYPE, FUNCTION>
                       (Result, A, B, C, exec, runflags, beginpoint, endpoint, true);
               }
           }
        }
    }
    else {
        ternary_op_guts<RET,ATYPE,BTYPE,CTYPE,FUNCTION> (Result, A, B, C, exec,
                                              runflags, beginpoint, endpoint, false);
    }
}


// Heavy lifting of the math and other binary ops, this is a templated
// version that knows the types of the arguments and the operation to
// perform (given by a functor).  This version can compute derivatives as long
// as FUNCTION has the required Dual2 operator implementations.
template <class RET, class ATYPE, class BTYPE, class FUNCTION>
inline void
binary_op_guts (Symbol &Result, Symbol &A, Symbol &B,
                ShadingExecution *exec, 
                Runflag *runflags, int beginpoint, int endpoint,
                bool zero_derivs=true)
{
    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, A.is_varying() | B.is_varying(),
                          A.data() == Result.data() || B.data() == Result.data());

    // Loop over points, do the operation
    VaryingRef<RET> result ((RET *)Result.data(), Result.step());
    VaryingRef<ATYPE> a ((ATYPE *)A.data(), A.step());
    VaryingRef<BTYPE> b ((BTYPE *)B.data(), B.step());
    FUNCTION function (exec);
    if (result.is_uniform()) {
        // Uniform case
        function (*result, *a, *b);
    } else if (A.is_uniform() && B.is_uniform()) {
        // Operands are uniform but we're assigning to a varying (it can
        // happen if we're in a conditional).  Take a shortcut by doing
        // the operation only once.
        RET r;
        function (r, *a, *b);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                function (result[i], a[i], b[i]);
    }
    if (zero_derivs && Result.has_derivs ())
        exec->zero_derivs (Result);
}

// Wrapper around binary_op_guts that has the call signature of an ordinary
// shadeop.
template <class RET, class ATYPE, class BTYPE, class FUNCTION>
DECLOP (binary_op_noderivs)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));

    binary_op_guts<RET,ATYPE,BTYPE,FUNCTION> (Result, A, B, exec,
                                              runflags, beginpoint, endpoint);
}



// Wrapper around binary_op_guts that does has he call signature of an
// ordinary shadeop, with support for derivatives.
// TODO: the presence of derivatives is static, but this shadeop checks
//       Symbol::has_derivs() everytime
template <class RET, class ATYPE, class BTYPE, class FUNCTION>
DECLOP (binary_op)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));

    if (Result.has_derivs()) {
        if (A.has_derivs()) {
            if (B.has_derivs())
                binary_op_guts<Dual2<RET>,Dual2<ATYPE>,Dual2<BTYPE>,FUNCTION> (Result, A, B, exec,
                                           runflags, beginpoint, endpoint, false);
            else
                binary_op_guts<Dual2<RET>,Dual2<ATYPE>,BTYPE,FUNCTION> (Result, A, B, exec,
                                           runflags, beginpoint, endpoint, false);
        } else if (B.has_derivs()) {
            binary_op_guts<Dual2<RET>,ATYPE,Dual2<BTYPE>,FUNCTION> (Result, A, B, exec,
                                           runflags, beginpoint, endpoint, false);
        } else {
            binary_op_guts<RET,ATYPE,BTYPE,FUNCTION> (Result, A, B, exec,
                                           runflags, beginpoint, endpoint,true);
        }
    } else {
        binary_op_guts<RET,ATYPE,BTYPE,FUNCTION> (Result, A, B, exec,
                                                  runflags, beginpoint, endpoint, false);
    }
}



// this is a binary function where only the "A" argument contributes to the
// result's derivatives
template <typename RET, typename ATYPE, typename BTYPE, typename FUNCTION>
DECLOP (binary_op_unary_derivs)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));

    if (Result.has_derivs() && A.has_derivs()) {
        binary_op_guts<Dual2<RET>,Dual2<ATYPE>,BTYPE,FUNCTION> (Result, A, B, exec,
                runflags, beginpoint, endpoint, false);
    } else {
        binary_op_guts<RET,ATYPE,BTYPE,FUNCTION> (Result, A, B, exec,
                runflags, beginpoint, endpoint, true);
    }
}



// Heavy lifting of the math and other unary ops, this is a templated
// version that knows the types of the arguments and the operation to
// perform (given by a functor).  This version is unaware of how to
// compute derivatives, so just clears them.
template <class RET, class ATYPE, class FUNCTION>
inline void
unary_op_guts_noderivs (Symbol &Result, Symbol &A,
                        ShadingExecution *exec, 
                        Runflag *runflags, int beginpoint, int endpoint)
{
    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, A.is_varying(), A.data() == Result.data());

    // FIXME -- clear derivs for now, make it right later.
    if (Result.has_derivs ())
        exec->zero_derivs (Result);

    // Loop over points, do the operation
    VaryingRef<RET> result ((RET *)Result.data(), Result.step());
    VaryingRef<ATYPE> a ((ATYPE *)A.data(), A.step());
    FUNCTION function (exec);
    if (result.is_uniform()) {
        // Uniform case
        function (*result, *a);
    } else if (A.is_uniform()) {
        // Operands are uniform but we're assigning to a varying (it can
        // happen if we're in a conditional).  Take a shortcut by doing
        // the operation only once.
        RET r;
        function (r, *a);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                function (result[i], a[i]);
    }
}



// Heavy lifting of the math and other unary ops, this is a templated
// version that knows the types of the arguments and the operation to
// perform (given by a functor).  This version computes derivatives.
template <class RET, class ATYPE, class FUNCTION>
inline void
unary_op_guts (Symbol &Result, Symbol &A,
               ShadingExecution *exec, 
               Runflag *runflags, int beginpoint, int endpoint)
{
    // Adjust the result's uniform/varying status
    exec->adjust_varying (Result, A.is_varying(), A.data() == Result.data());

    // Loop over points, do the operation
    FUNCTION function (exec);
    if (Result.is_uniform()) {
        // Uniform case
        function (*((RET *)Result.data()), *(ATYPE *)A.data());
        if (Result.has_derivs())
            exec->zero_derivs (Result);
    } else if (A.is_uniform()) {
        // Operands are uniform but we're assigning to a varying (it can
        // happen if we're in a conditional).  Take a shortcut by doing
        // the operation only once.
        RET r;
        function (r, *(ATYPE *)A.data());
        VaryingRef<RET> result ((RET *)Result.data(), Result.step());
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = r;
        if (Result.has_derivs())
            exec->zero_derivs (Result);
    } else {
        // Fully varying case
        if (Result.has_derivs() && A.has_derivs()) {
            VaryingRef<Dual2<RET> > result ((Dual2<RET> *)Result.data(), Result.step());
            VaryingRef<Dual2<ATYPE> > a ((Dual2<ATYPE> *)A.data(), A.step());
            for (int i = beginpoint;  i < endpoint;  ++i)
                if (runflags[i])
                    function (result[i], a[i]);
        } else {
            VaryingRef<RET> result ((RET *)Result.data(), Result.step());
            VaryingRef<ATYPE> a ((ATYPE *)A.data(), A.step());
            for (int i = beginpoint;  i < endpoint;  ++i)
                if (runflags[i])
                    function (result[i], a[i]);
            if (Result.has_derivs())
                exec->zero_derivs (Result);
        }
    }
}



// Heavy lifting of the math and other unary ops, this is a templated
// version that knows the types of the arguments and the operation to
// perform (given by a functor).
template <class RET, class ATYPE, class FUNCTION>
DECLOP (unary_op_noderivs)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));

    unary_op_guts_noderivs<RET,ATYPE,FUNCTION> (Result, A, exec,
                                                runflags, beginpoint, endpoint);
}


template <class RET, class ATYPE, class FUNCTION>
DECLOP (unary_op)
{
    // Get references to the symbols this op accesses
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));

    unary_op_guts<RET,ATYPE,FUNCTION> (Result, A, exec,
                                       runflags, beginpoint, endpoint);
}


/// Implements the opcode for a specific ClosurePrimitive in the "standard way
template <typename Primitive, int NumArgs> inline
DECLOP (closure_op_guts)
{
    ASSERT (nargs >= NumArgs); // TODO: switch to DASSERT at some point

    Symbol &Result (exec->sym (args[0]));
    DASSERT(Result.typespec().is_closure() && Result.is_varying());

    /* try to parse token/values pair (if there are any) */
    VaryingRef<ustring> sidedness(NULL, 0);
    VaryingRef<ustring> labels[ClosurePrimitive::MAXCUSTOM+1];
    int nlabels = 0;
    for (int tok = NumArgs; tok < nargs; tok += 2) {
        Symbol &Name (exec->sym (args[tok]));
        DASSERT (Name.typespec().is_string() && "optional closure token must be a string");
        DASSERT (tok + 1 < nargs && "malformed argument list for closure");
        ustring name = * (ustring *) Name.data();
        Symbol &Val (exec->sym (args[tok + 1]));
        if (name == Strings::sidedness && Val.typespec().is_string()) {
            sidedness.init((ustring*) Val.data(), Val.step());
        } else if (name == Strings::label && Val.typespec().is_string()) {
            if (nlabels == ClosurePrimitive::MAXCUSTOM)
                exec->error ("Too many labels to closure (%s:%d)",
                                     exec->op().sourcefile().c_str(),
                                     exec->op().sourceline());
            else {
               labels[nlabels].init((ustring*) Val.data(), Val.step());
               nlabels++;
            }
        } else {
            exec->error ("Unknown closure optional argument: \"%s\", <%s> (%s:%d)",
                                     name.c_str(),
                                     Val.typespec().c_str(),
                                     exec->op().sourcefile().c_str(),
                                     exec->op().sourceline());
        }
    }

    /* N.B. Closures don't have derivs */
    VaryingRef<ClosureColor *> result ((ClosureColor **)Result.data(), Result.step());
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            char* mem = result[i]->allocate_component (sizeof (Primitive));
            ClosurePrimitive::Sidedness side = ClosurePrimitive::Front;
            if (sidedness) {
                if (sidedness[i] == Strings::front)
                    side = ClosurePrimitive::Front;
                else if (sidedness[i] == Strings::back)
                    side = ClosurePrimitive::Back;
                else if (sidedness[i] == Strings::both)
                    side = ClosurePrimitive::Both;
                else
                    side = ClosurePrimitive::None;
            }
            ClosurePrimitive *prim = new (mem) Primitive (i, exec, nargs, args, side);
            for (int l = 0; l < nlabels; ++l)
               prim->set_custom_label(l, labels[l][i]);
            // Label list must be NONE terminated
            prim->set_custom_label(nlabels, Labels::NONE);
        }
    }
}

/// Fetch the value of an opcode argument given its index in the arglist
template <typename T> inline
void fetch_value (T &v, int argidx, int idx, ShadingExecution *exec, int nargs, const int *args)
{
    DASSERT (argidx < nargs);
    // TODO: typecheck assert?
    Symbol &Sym = exec->sym (args[argidx]);
    VaryingRef<T> values ((T*) Sym.data(), Sym.step());
    v = values[idx];
}

/// Standard form for a closure constructor
#define CLOSURE_CTOR(name)              \
    name (int idx, ShadingExecution *exec, int nargs, const int *args, Sidedness side)

/// Helper macros to extract values from the opcopde argument list
#define CLOSURE_FETCH_ARG(v, argidx)    \
    fetch_value(v, argidx, idx, exec, nargs, args)


// Proxy type that derives from Vec3 but allows some additional operations
// not normally supported by Imath::Vec3.  This is purely for convenience.
class VecProxy : public Vec3 {
public:
    VecProxy () { }
    VecProxy (float a) : Vec3(a,a,a) { }
    VecProxy (float a, float b, float c) : Vec3(a,b,c) { }
    VecProxy (const Vec3& v) : Vec3(v) { }

    friend VecProxy operator+ (const Vec3 &v, float f) {
        return VecProxy (v.x+f, v.y+f, v.z+f);
    }
    friend VecProxy operator+ (float f, const Vec3 &v) {
        return VecProxy (v.x+f, v.y+f, v.z+f);
    }
    friend VecProxy operator- (const Vec3 &v, float f) {
        return VecProxy (v.x-f, v.y-f, v.z-f);
    }
    friend VecProxy operator- (float f, const Vec3 &v) {
        return VecProxy (f-v.x, f-v.y, f-v.z);
    }
    friend VecProxy operator* (const Vec3 &v, int f) {
        return VecProxy (v.x*f, v.y*f, v.z*f);
    }
    friend VecProxy operator* (int f, const Vec3 &v) {
        return VecProxy (v.x*f, v.y*f, v.z*f);
    }
    friend VecProxy operator/ (const Vec3 &v, int f) {
        if (f == 0)
            return VecProxy(0.0);
        return VecProxy (v.x/f, v.y/f, v.z/f);
    }
    friend VecProxy operator/ (float f, const Vec3 &v) {
        return VecProxy (v.x == 0.0 ? 0.0 : f/v.x, 
                         v.y == 0.0 ? 0.0 : f/v.y,
                         v.z == 0.0 ? 0.0 : f/v.z);
    }
    friend VecProxy operator/ (int f, const Vec3 &v) {
        return VecProxy (v.x == 0.0 ? 0.0 : f/v.x, 
                         v.y == 0.0 ? 0.0 : f/v.y,
                         v.z == 0.0 ? 0.0 : f/v.z);
    }
    friend bool operator== (const Vec3 &v, float f) {
        return v.x == f && v.y == f && v.z == f;
    }
    friend bool operator== (const Vec3 &v, int f) {
        return v.x == f && v.y == f && v.z == f;
    }
    friend bool operator== (float f, const Vec3 &v) {
        return v.x == f && v.y == f && v.z == f;
    }
    friend bool operator== (int f, const Vec3 &v) {
        return v.x == f && v.y == f && v.z == f;
    }

    friend bool operator!= (const Vec3 &v, float f) {
        return v.x != f || v.y != f || v.z != f;
    }
    friend bool operator!= (const Vec3 &v, int f) {
        return v.x != f || v.y != f || v.z != f;
    }
    friend bool operator!= (float f, const Vec3 &v) {
        return v.x != f || v.y != f || v.z != f;
    }
    friend bool operator!= (int f, const Vec3 &v) {
        return v.x != f || v.y != f || v.z != f;
    }
};



// Proxy type that derives from Matrix44 but allows assignment of a float
// to mean f*Identity.
class MatrixProxy : public Matrix44 {
public:
    MatrixProxy () { }
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

    friend bool operator== (const MatrixProxy &m, float f) {
        MatrixProxy comp (f);
        return m == comp;
    }
    friend bool operator== (const MatrixProxy &m, int f) {
        MatrixProxy comp (f);
        return m == comp;
    }
    friend bool operator== (float f, const MatrixProxy &m) { return m == f; }
    friend bool operator== (int f, const MatrixProxy &m) { return m == f; }

    friend bool operator!= (const MatrixProxy &m, float f) {
        MatrixProxy comp (f);
        return m != comp;
    }
    friend bool operator!= (const MatrixProxy &m, int f) {
        MatrixProxy comp (f);
        return m != comp;
    }
    friend bool operator!= (float f, const MatrixProxy &m) { return m != f; }
    friend bool operator!= (int f, const MatrixProxy &m) { return m != f; }
};



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSLOPS_H */
