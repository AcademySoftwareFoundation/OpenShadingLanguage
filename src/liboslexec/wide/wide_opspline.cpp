// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of spline
/// operator
///
/////////////////////////////////////////////////////////////////////////

#include <OSL/oslconfig.h>

#include "oslexec_pvt.h"

#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/sfm_staticmatrix.h>
#include <OSL/sfmath.h>
#include <OSL/wide.h>

#include "splineimpl.h"

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"

//using namespace std;

// non SIMD version, should be scalar code meant to be used
// inside SIMD loops
// SIMD FRIENDLY MATH
namespace sfm {

template<class K_T, bool IsBasisUConstantT, int BasisStepT, class MatrixT,
         class R_T, class X_T, class KArrayT>
OSL_FORCEINLINE void
spline_weighted_evaluate(const MatrixT& M, R_T& result, X_T& xval,
                         KArrayT knots, int knot_count)
{
    using OIIO::clamp;
    X_T x       = clamp(xval, X_T(0.0), X_T(1.0));
    int nsegs   = ((knot_count - 4) / BasisStepT) + 1;
    X_T x2      = x * (float)nsegs;
    float seg_x = removeDerivatives(x2);
    int segnum  = (int)seg_x;
    if (segnum < 0)
        segnum = 0;
    if (segnum > (nsegs - 1))
        segnum = nsegs - 1;

    if (IsBasisUConstantT) {
        // Special case for "constant" basis
        K_T constantKnot = knots[segnum + 1];
        R_T P            = removeDerivatives(constantKnot);
        assignment(result, P);
        return;
    }
    // x is the position along segment 'segnum'
    X_T x3 = x2 - float(segnum);
    int s  = segnum * BasisStepT;

    // extract the knot elements

    K_T P0 = knots[s];
    K_T P1 = knots[s + 1];
    K_T P2 = knots[s + 2];
    K_T P3 = knots[s + 3];
    // std::cout << "seg knot[0] = " << P0 << std::endl;
    // std::cout << "seg knot[1] = " << P1 << std::endl;
    // std::cout << "seg knot[2] = " << P2 << std::endl;
    // std::cout << "seg knot[3] = " << P3 << std::endl;

    auto tk0 = M.m00 * P0 + M.m01 * P1 + M.m02 * P2 + M.m03 * P3;

    auto tk1 = M.m10 * P0 + M.m11 * P1 + M.m12 * P2 + M.m13 * P3;

    auto tk2 = M.m20 * P0 + M.m21 * P1 + M.m22 * P2 + M.m23 * P3;

    auto tk3 = M.m30 * P0 + M.m31 * P1 + M.m32 * P2 + M.m33 * P3;

    R_T tresult = sfm::unproxy_element(((tk0 * x3 + tk1) * x3 + tk2) * x3
                                       + tk3);
    // std::cout << "tresult=" << tresult << std::endl;

    assignment(result, tresult);
}



// Derived version from OIIO::invert released under the same
// PDX-License-Identifier: BSD-3-Clause

/// Solve for the x for which func(x) == y on the interval [xmin,xmax].
/// Use a maximum of maxiter iterations, and stop any time the remaining
/// search interval or the function evaluations <= eps.
/// is_bracketed_by(xmin, xmax) returns true if y is in [f(xmin), f(xmax)],
/// otherwise false (in which case the caller should know that the results
/// may be unreliable.  A 2nd call to bracketed_invert(xmin, xmax) is then
/// required to search within that bracketed span.  The test for bracketing
/// is separated out so the calling code can exit the knot iteration loop,
/// before calling bracketed_invert (which has its own loop).  The separation
/// allows the avoidance of a nested loop which would have lower coherency
/// under SIMD/SIMT execution than 2 separate loops.
/// Results are undefined if the function is not monotonic
/// on that interval or if there are multiple roots in the interval (it
/// may not converge, or may converge to any of the roots without
/// telling you that there are more than one).
template<class T, class Func, int maxiters = 32> class Inverter {
    Func& m_func;
    const T& m_y;
    const T m_eps;
    T m_result;
    T m_v0;
    T m_v1;
    bool m_increasing;

public:
    Inverter(const T& initial_result, Func& func, const T& y, const T& eps)
        : m_func(func), m_y(y), m_eps(eps), m_result(initial_result)
    {
    }

    const T& result() const { return m_result; }

    OSL_FORCEINLINE
    bool is_bracketed_by(const T& xmin, const T& xmax)
    {
        using ::fabs;
        // Use the Regula Falsi method, falling back to bisection if it
        // hasn't converged after 3/4 of the maximum number of iterations.
        // See, e.g., Numerical Recipes for the basic ideas behind both
        // methods.
        m_v0         = m_func(xmin);
        m_v1         = m_func(xmax);
        m_increasing = (m_v0 < m_v1);
#if 0
        // ternary was using pointer to m_v0 or m_v1 which disallows privatization
        // of their data layouts, causing lots of strided memory stores
        T vmin = m_increasing ? m_v0 : m_v1;
        T vmax = m_increasing ? m_v1 : m_v0;
#else
        // Instead make sure only values are used, no pointers which
        // enables Scalar Replacement of Aggregates avoiding memory stores
        T vmin = sfm::select_val(m_increasing, m_v0, m_v1);
        T vmax = sfm::select_val(m_increasing, m_v1, m_v0);
#endif
        // To simply control flow for vectorizor, changed logical &&
        // to bitwise &.  This is also preferable to minimize extra
        // masking and potential branching
        bool brack = ((m_y >= vmin) & (m_y <= vmax));
        if (!brack) {
            // If our bounds don't bracket the zero, just give up, and
            // return the appropriate "edge" of the interval
#if 0
            // ternary was using pointer to m_v0 or m_v1 which disallows privatization
            // of their data layouts, causing lots of strided memory stores
            m_result = ((m_y < vmin) == m_increasing) ? xmin : xmax;
#else
            // Instead make sure only values are used, no pointers which
            // enables Scalar Replacement of Aggregates avoiding memory stores
            m_result = sfm::select_val(((m_y < vmin) == m_increasing), xmin,
                                       xmax);
#endif
        }
        return brack;
    }

    OSL_FORCEINLINE
    void bracketed_invert(T xmin, T xmax)
    {
        // Assumes that is_bracketed_by(xmin, xmax) was called and returned true
        using ::fabs;

        m_result = xmin;
        if (fabs(m_v0 - m_v1) < m_eps) {  // already close enough
            return;
        }

        constexpr int rfiters = (3 * maxiters)
                                / 4;  // how many times to try regula falsi
        for (int iters = 0; iters < maxiters; ++iters) {
            T t;  // interpolation factor
            if (iters < rfiters) {
                // Regula falsi
                t = (m_y - m_v0) / (m_v1 - m_v0);
                // To simply control flow for vectorizor, changed logical ||
                // to bitwise |.  This is also preferable to minimize extra
                // masking and potential branching
                if ((t <= T(0)) | (t >= T(1)))
                    t = T(0.5);  // RF convergence failure -- bisect instead
            } else {
                t = T(0.5);  // bisection
            }
            m_result = OIIO::lerp(xmin, xmax, t);
            T v      = m_func(m_result);
            if ((v < m_y) == m_increasing) {
                xmin = m_result;
                m_v0 = v;
            } else {
                xmax = m_result;
                m_v1 = v;
            }
            // To simply control flow for vectorizor, changed logical ||
            // to bitwise |.  This is also preferable to minimize extra
            // masking and potential branching
            if ((fabs(xmax - xmin) < m_eps) | (fabs(v - m_y) < m_eps))
                return;  // converged
        }
    }
};



// Spline functor for use with the inverse function
template<class K_T, bool IsBasisUConstantT, int BasisStepT, class MatrixT,
         class R_T, class X_T, class KArrayT>
struct SplineSearchFunctor {
    SplineSearchFunctor(const MatrixT& M_, KArrayT knots_, int knot_count_)
        : M(M_), knots(knots_), knot_count(knot_count_)
    {
    }

    R_T operator()(X_T x)
    {
        R_T v;
        sfm::spline_weighted_evaluate<K_T, IsBasisUConstantT, BasisStepT>(
            M, v, x, knots, knot_count);
        return v;
    }

private:
    const MatrixT& M;
    // copy knots array by value,  because it's a proxy that
    // only holds reference/pointer to actual data.
    // NOTE: arraylength is embedded in the knots proxy
    KArrayT knots;
    int knot_count;
};

// NOTE: keep implementation in synch with SplineInterp::inverse
template<class K_T, bool IsBasisUConstantT, int BasisStepT, class MatrixT,
         class R_T, class X_T, class KArrayT>
OSL_FORCEINLINE void
splineinverse_search(const MatrixT& M, R_T& result, X_T& xval, KArrayT knots,
                     int knot_count)
{
    // account for out-of-range inputs, just clamp to the values we have
    // NOTE: ternary is based on compile time known BasisStepT
    // so not conditional should exist after dead code elimination
    constexpr int lowindex = (BasisStepT == 1) ? 1 : 0;
    int highindex = (BasisStepT == 1) ? (knot_count - 2) : (knot_count - 1);

    K_T lowKnot  = knots[lowindex];
    K_T highKnot = knots[highindex];
#if 0  // Reference version
    // When vectorized, control flow produced some incorrect derivative
    // results for splines with constant basis

    //bool increasing = knots[1] < knots[knot_count-2];
    K_T k1 = knots[1]; // knots is a proxy, we must pull value out to local
    K_T k2 = knots[knot_count-2];
    bool increasing = k1 < k2;
    if (increasing) {
        if (xval <= lowKnot) {
            result = R_T(0);
            return;
        }
        if (xval >= highKnot) {
            result = R_T(1);
            return;
        }
    } else {
        if (xval >= lowKnot) {
            result = R_T(0);
            return;
        }
        if (xval <= highKnot) {
            result = R_T(1);
            return;
        }
    }
#else
    //bool increasing = knots[1] < knots[knot_count-2];
    bool increasing;
    if (BasisStepT == 1) {
        static_assert((lowindex == 1) || (BasisStepT != 1),
                      "unexpected lowindex");
        OSL_DASSERT(highindex == knot_count - 2);
        // We can directly use low and high knot values
        // versus dereferencing knots again to detect if we are increasing
        increasing = lowKnot < highKnot;
    } else {
        static_assert((lowindex == 0) || (BasisStepT == 1),
                      "unexpected lowindex");
        // We can NOT use the low and high knot values,
        // so will need to extract 2 more knot values
        // to detect if we are increasing
        K_T k1     = knots[1];  // knots is proxy, must export to local
        K_T k2     = knots[knot_count - 2];
        increasing = k1 < k2;
    }

    if ((increasing & (xval <= lowKnot))
        | ((!increasing) & (xval >= lowKnot))) {
        result = R_T(0);
        return;
    }
    if ((increasing & (xval >= highKnot))
        | ((!increasing) & (xval <= highKnot))) {
        result = R_T(1);
        return;
    }
#endif
    typedef SplineSearchFunctor<K_T, IsBasisUConstantT, BasisStepT, MatrixT,
                                R_T, X_T, KArrayT>
        Functor;
    Functor S(M, knots, knot_count);

    // NOTE: OIIO::invert has a loop which was called from inside the search
    // interval loop, created a nested loop.  Under SIMD/SIMT the nested
    // loop could execute incoherently.  Instead we choose to use the class
    // sfm::Inverter to manage state between is_bracketed_by(min,max), meant to
    // be called from the search interval loop, and bracketed_invert(min,max)
    // meant to be called after exiting the search interval loop, avoiding
    // a nested loop that could be executed at different interval's
    // for each SIMD/SIMT lane/thread.
    sfm::Inverter<X_T, Functor, /*maxiters=*/32> inverter {
        /*initial_result=*/0, S, xval, X_T(1.0e-6)
    };

    // Because of the nature of spline interpolation, monotonic knots
    // can still lead to a non-monotonic curve.  To deal with this,
    // search separately on each spline segment and hope for the best.
    int nsegs     = (knot_count - 4) / BasisStepT + 1;
    float nseginv = 1.0f / nsegs;
    X_T r0        = 0.0;
    X_T r1;
    bool bracket_found = false;
    for (int s = 0; s < nsegs; ++s) {  // Search each interval
        r1            = nseginv * (s + 1);
        bracket_found = inverter.is_bracketed_by(r0, r1);
        if (bracket_found)
            break;
        r0 = r1;  // Start of next interval is end of this one
    }

    if (bracket_found) {
        // NOTE: do not call bracketed_invert
        // from inside the search interval loop
        inverter.bracketed_invert(r0, r1);
    }
    result = inverter.result();
}

}  // namespace sfm


namespace {  // unnamed

template<bool IsBasisUConstantT, int BasisStepT, typename MatrixT,
         typename RAccessorT, typename XAccessorT, typename KAccessorT>
OSL_FORCEINLINE void
spline_evaluate_loop_over_wide(const MatrixT& M, RAccessorT wR, XAccessorT wX,
                               KAccessorT wK, int knot_count)
{
    static constexpr int vec_width = RAccessorT::width;

    typedef typename XAccessorT::NonConstValueType X_Type;
    typedef typename RAccessorT::ValueType R_Type;
    typedef typename KAccessorT::ElementType K_Type;

    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(vec_width))
        for (int lane = 0; lane < vec_width; ++lane) {
            X_Type x   = wX[lane];
            auto knots = wK[lane];

            if (wR.mask()[lane]) {
                R_Type result;
                sfm::spline_weighted_evaluate<K_Type, IsBasisUConstantT,
                                              BasisStepT>(M, result, x, knots,
                                                          knot_count);
                wR[ActiveLane(lane)] = result;
            }
        }
    }
}


typedef sfm::StaticMatrix44<-1, 3, -3, 1, 2, -5, 4, -1, -1, 0, 1, 0, 0, 2, 0, 0,
                            2 /* divisor */>
    CatmullRomWeights;
typedef sfm::StaticMatrix44<-1, 3, -3, 1, 3, -6, 3, 0, -3, 3, 0, 0, 1, 0, 0, 0,
                            1 /*divisor*/>
    BezierWeights;
typedef sfm::StaticMatrix44<-1, 3, -3, 1, 3, -6, 3, 0, -3, 0, 3, 0, 1, 4, 1, 0,
                            6 /*divisor*/>
    BsplineWeights;
typedef sfm::StaticMatrix44<2, 1, -2, 1, -3, -2, 3, -1, 0, 1, 0, 0, 1, 0, 0, 0,
                            1 /*divisor*/>
    HermiteWeights;
typedef sfm::StaticMatrix44<0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, 0, 0,
                            1 /*divisor*/>
    LinearWeights;
// NOTE:  when basis is constant the weights are ignored,
// just pass in 0's for the compiler to ignore
typedef sfm::StaticMatrix44<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1 /*Divisor*/>
    ConstantWeights;



template<bool IsBasisUConstantT, int BasisStepT, typename MatrixT,
         typename RAccessorT, typename XAccessorT, typename KAccessorT>
static OSL_NOINLINE void
spline_evaluate_wide_with_static_matrix(RAccessorT wR, XAccessorT wX,
                                        KAccessorT wK, int knot_count)
{
    MatrixT m;
    spline_evaluate_loop_over_wide<IsBasisUConstantT, BasisStepT>(m, wR, wX, wK,
                                                                  knot_count);
}



template<typename RAccessorT, typename XAccessorT, typename KAccessor_T>
OSL_FORCEINLINE void
spline_evaluate_wide(RAccessorT wR, ustring spline_basis, XAccessorT wX,
                     KAccessor_T wK, int knot_count)
{
    typedef void (*FuncPtr)(RAccessorT, XAccessorT, KAccessor_T,
                            int /*knot_count*/);
    static constexpr FuncPtr impl_by_basis[6]
        = { &spline_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 1 /* basis_step */,
                CatmullRomWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &spline_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 3 /* basis_step */,
                BezierWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &spline_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 1 /* basis_step */,
                BsplineWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &spline_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 2 /* basis_step */,
                HermiteWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &spline_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 1 /* basis_step */,
                LinearWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &spline_evaluate_wide_with_static_matrix<
                true /*is_basis_u_constant */, 1 /* basis_step */,
                ConstantWeights, RAccessorT, XAccessorT, KAccessor_T> };

    unsigned int basis_type = Spline::basis_type_of(spline_basis);

    OSL_DASSERT(basis_type < Spline::kNumSplineTypes
                && "unsupported spline basis");

    impl_by_basis[basis_type](wR, wX, wK, knot_count);
}



template<bool IsBasisUConstantT, int BasisStepT, typename MatrixT,
         typename RAccessorT, typename XAccessorT, typename KAccessorT>
OSL_FORCEINLINE void
splineinverse_evaluate_loop_over_wide(const MatrixT& M, RAccessorT wR,
                                      XAccessorT wX, KAccessorT wK,
                                      int knot_count)
{
    static constexpr int vec_width = RAccessorT::width;

    typedef typename XAccessorT::NonConstValueType X_Type;
    typedef typename RAccessorT::ValueType R_Type;
    typedef typename KAccessorT::ElementType K_Type;

    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_COMPLEX_SIMD_LOOP(simdlen(vec_width))
        for (int lane = 0; lane < vec_width; ++lane) {
            X_Type x   = wX[lane];
            auto knots = wK[lane];

            if (wR.mask()[lane]) {
                R_Type result;
                sfm::splineinverse_search<K_Type, IsBasisUConstantT, BasisStepT>(
                    M, result, x, knots, knot_count);

                wR[ActiveLane(lane)] = result;
            }
        }
    }
}



template<bool IsBasisUConstantT, int BasisStepT, typename MatrixT,
         typename RAccessorT, typename XAccessorT, typename KAccessorT>
static OSL_NOINLINE void
splineinverse_evaluate_wide_with_static_matrix(RAccessorT wR, XAccessorT wX,
                                               KAccessorT wK, int knot_count)
{
    MatrixT m;
    splineinverse_evaluate_loop_over_wide<IsBasisUConstantT, BasisStepT>(
        m, wR, wX, wK, knot_count);
}



template<typename RAccessorT, typename XAccessorT, typename KAccessor_T>
OSL_FORCEINLINE void
splineinverse_evaluate_wide(RAccessorT wR, ustring spline_basis, XAccessorT wX,
                            KAccessor_T wK, int knot_count)
{
    typedef void (*FuncPtr)(RAccessorT, XAccessorT, KAccessor_T,
                            int /*knot_count*/);
    static constexpr FuncPtr impl_by_basis[6]
        = { &splineinverse_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 1 /* basis_step */,
                CatmullRomWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &splineinverse_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 3 /* basis_step */,
                BezierWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &splineinverse_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 1 /* basis_step */,
                BsplineWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &splineinverse_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 2 /* basis_step */,
                HermiteWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &splineinverse_evaluate_wide_with_static_matrix<
                false /*is_basis_u_constant */, 1 /* basis_step */,
                LinearWeights, RAccessorT, XAccessorT, KAccessor_T>,
            &splineinverse_evaluate_wide_with_static_matrix<
                true /*is_basis_u_constant */, 1 /* basis_step */,
                ConstantWeights, RAccessorT, XAccessorT, KAccessor_T> };

    unsigned int basis_type = Spline::basis_type_of(spline_basis);

    OSL_DASSERT(basis_type < Spline::kNumSplineTypes
                && "unsupported spline basis");

    impl_by_basis[basis_type](wR, wX, wK, knot_count);
}

}  // namespace


// TODO:  would be better if the code generator called a specific version of
// spline (hermite, bezier, etc.) vs. making us doing a bunch of comparisons

OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wf, Wf, f)(void* wout_, const char* spline_, void* wx_,
                                    float* knots, int knot_count,
                                    int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<float>(wout_, Mask(mask_value)), USTR(spline_),
                         Wide<const float>(wx_),
                         UniformAsWide<const float[]>(knots, knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wf, f, Wf)(void* wout_, const char* spline_, void* wx_,
                                    float* knots, int knot_count,
                                    int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<float>(wout_, Mask(mask_value)), USTR(spline_),
                         UniformAsWide<const float>(wx_),
                         Wide<const float[]>(knots, knot_arraylen), knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdf, Wdf, Wdf)(void* wout_, const char* spline_,
                                        void* wx_, float* knots, int knot_count,
                                        int knot_arraylen,
                                        unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<float>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const Dual2<float>>(wx_),
                         Wide<const Dual2<float>[]>(knots, knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdf, Wdf,
                 df)(void* wout_, const char* spline_, void* wx_, float* knots,
                     int knot_count, int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<float>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const Dual2<float>>(wx_),
                         UniformAsWide<const Dual2<float>[]>(knots,
                                                             knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdf, Wf,
                 df)(void* wout_, const char* spline_, void* wx_, float* knots,
                     int knot_count, int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<float>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const float>(wx_),
                         UniformAsWide<const Dual2<float>[]>(knots,
                                                             knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdf, df, Wdf)(void* wout_, const char* spline_,
                                       void* wx_, float* knots, int knot_count,
                                       int knot_arraylen,
                                       unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<float>>(wout_, Mask(mask_value)),
                         USTR(spline_), UniformAsWide<const Dual2<float>>(wx_),
                         Wide<const Dual2<float>[]>(knots, knot_arraylen),
                         knot_count);
}



//===========================================================================

OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdf, f, Wdf)(void* wout_, const char* spline_,
                                      void* wx_, float* knots, int knot_count,
                                      int knot_arraylen,
                                      unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<float>>(wout_, Mask(mask_value)),
                         USTR(spline_), UniformAsWide<const float>(wx_),
                         Wide<const Dual2<float>[]>(knots, knot_arraylen),
                         knot_count);
}


OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdf, Wf, Wdf)(void* wout_, const char* spline_,
                                       void* wx_, float* knots, int knot_count,
                                       int knot_arraylen,
                                       unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<float>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const float>(wx_),
                         Wide<const Dual2<float>[]>(knots, knot_arraylen),
                         knot_count);
}



//===========================================================================
OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdf, Wdf,
                 f)(void* wout_, const char* spline_, void* wx_, float* knots,
                    int knot_count, int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<float>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const Dual2<float>>(wx_),
                         UniformAsWide<const float[]>(knots, knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wf, Wf, Wf)(void* wout_, const char* spline_,
                                     void* wx_, void* wknots_, int knot_count,
                                     int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<float>(wout_, Mask(mask_value)), USTR(spline_),
                         Wide<const float>(wx_),
                         Wide<const float[]>(wknots_, knot_arraylen),
                         knot_count);
}



//=======================================================================
OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wv, Wf, v)(void* wout_, const char* spline_, void* wx_,
                                    Vec3* knots, int knot_count,
                                    int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Vec3>(wout_, Mask(mask_value)), USTR(spline_),
                         Wide<const float>(wx_),
                         UniformAsWide<const Vec3[]>(knots, knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wv, Wf, Wv)(void* wout_, const char* spline_,
                                     void* wx_, Vec3* knots, int knot_count,
                                     int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Vec3>(wout_, Mask(mask_value)), USTR(spline_),
                         Wide<const float>(wx_),
                         Wide<const Vec3[]>(knots, knot_arraylen), knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wv, f, Wv)(void* wout_, const char* spline_, void* wx_,
                                    Vec3* knots, int knot_count,
                                    int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Vec3>(wout_, Mask(mask_value)), USTR(spline_),
                         UniformAsWide<const float>(wx_),
                         Wide<const Vec3[]>(knots, knot_arraylen), knot_count);
}

//=======================================================================

OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdv, Wdf,
                 v)(void* wout_, const char* spline_, void* wx_, Vec3* knots,
                    int knot_count, int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<Vec3>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const Dual2<float>>(wx_),
                         UniformAsWide<const Vec3[]>(knots, knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdv, Wdf,
                 Wv)(void* wout_, const char* spline_, void* wx_, Vec3* knots,
                     int knot_count, int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<Vec3>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const Dual2<float>>(wx_),
                         Wide<const Vec3[]>(knots, knot_arraylen), knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdv, df,
                 Wv)(void* wout_, const char* spline_, void* wx_, Vec3* knots,
                     int knot_count, int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<Vec3>>(wout_, Mask(mask_value)),
                         USTR(spline_), UniformAsWide<const Dual2<float>>(wx_),
                         Wide<const Vec3[]>(knots, knot_arraylen), knot_count);
}



//=======================================================================
OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdv, f, Wdv)(void* wout_, const char* spline_,
                                      void* wx_, Vec3* knots, int knot_count,
                                      int knot_arraylen,
                                      unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<Vec3>>(wout_, Mask(mask_value)),
                         USTR(spline_), UniformAsWide<const float>(wx_),
                         Wide<const Dual2<Vec3>[]>(knots, knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdv, Wf, Wdv)(void* wout_, const char* spline_,
                                       void* wx_, Vec3* knots, int knot_count,
                                       int knot_arraylen,
                                       unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<Vec3>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const float>(wx_),
                         Wide<const Dual2<Vec3>[]>(knots, knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdv, Wf,
                 dv)(void* wout_, const char* spline_, void* wx_, Vec3* knots,
                     int knot_count, int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<Vec3>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const float>(wx_),
                         UniformAsWide<const Dual2<Vec3>[]>(knots,
                                                            knot_arraylen),
                         knot_count);
}



//=======================================================================
OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdv, Wdf, Wdv)(void* wout_, const char* spline_,
                                        void* wx_, Vec3* knots, int knot_count,
                                        int knot_arraylen,
                                        unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<Vec3>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const Dual2<float>>(wx_),
                         Wide<const Dual2<Vec3>[]>(knots, knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdv, Wdf,
                 dv)(void* wout_, const char* spline_, void* wx_, Vec3* knots,
                     int knot_count, int knot_arraylen, unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<Vec3>>(wout_, Mask(mask_value)),
                         USTR(spline_), Wide<const Dual2<float>>(wx_),
                         UniformAsWide<const Dual2<Vec3>[]>(knots,
                                                            knot_arraylen),
                         knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(spline, Wdv, df, Wdv)(void* wout_, const char* spline_,
                                       void* wx_, Vec3* knots, int knot_count,
                                       int knot_arraylen,
                                       unsigned int mask_value)
{
    spline_evaluate_wide(Masked<Dual2<Vec3>>(wout_, Mask(mask_value)),
                         USTR(spline_), UniformAsWide<const Dual2<float>>(wx_),
                         Wide<const Dual2<Vec3>[]>(knots, knot_arraylen),
                         knot_count);
}


//=======================================================================
// TODO:  WIDE spline inverse currently operates in scalar working over each
// active lane.  There is room for improvement, even in the scalar version.
// An alternative version of Spline::spline_inverse could be made that
// calls our spline_evaluate_scalar, better yet chooses a specialization
// based on the spline basis type which should be uniform.
// TODO:  After the above optimization a SIMD version could be attempted.


OSL_BATCHOP void
__OSL_MASKED_OP3(splineinverse, Wf, Wf,
                 Wf)(void* wout_, const char* spline_, void* wx_, void* wknots_,
                     int knot_count, int knot_arraylen, unsigned int mask_value)
{
    // Version with no derivs
    splineinverse_evaluate_wide(Masked<float>(wout_, Mask(mask_value)),
                                USTR(spline_), Wide<const float>(wx_),
                                Wide<const float[]>(wknots_, knot_arraylen),
                                knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(splineinverse, Wf, Wf,
                 f)(void* wout_, const char* spline_, void* wx_, void* wknots_,
                    int knot_count, int knot_arraylen, unsigned int mask_value)
{
    // Version with no derivs
    splineinverse_evaluate_wide(Masked<float>(wout_, Mask(mask_value)),
                                USTR(spline_), Wide<const float>(wx_),
                                UniformAsWide<const float[]>(wknots_,
                                                             knot_arraylen),
                                knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(splineinverse, Wf, f,
                 Wf)(void* wout_, const char* spline_, void* wx_, void* wknots_,
                     int knot_count, int knot_arraylen, unsigned int mask_value)
{
    // Version with no derivs
    splineinverse_evaluate_wide(Masked<float>(wout_, Mask(mask_value)),
                                USTR(spline_), UniformAsWide<const float>(wx_),
                                Wide<const float[]>(wknots_, knot_arraylen),
                                knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(splineinverse, Wdf, Wdf,
                 f)(void* wout_, const char* spline_, void* wx_, void* wknots_,
                    int knot_count, int knot_arraylen, unsigned int mask_value)
{
    // x has derivs, so return derivs as well
    splineinverse_evaluate_wide(Masked<Dual2<float>>(wout_, Mask(mask_value)),
                                USTR(spline_), Wide<const Dual2<float>>(wx_),
                                UniformAsWide<const float[]>(wknots_,
                                                             knot_arraylen),
                                knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(splineinverse, Wdf, Wdf,
                 Wdf)(void* wout_, const char* spline_, void* wx_,
                      void* wknots_, int knot_count, int knot_arraylen,
                      unsigned int mask_value)
{
    // Ignore knot derivatives
    //
    // x has derivs, so return derivs as well
    splineinverse_evaluate_wide(
        Masked<Dual2<float>>(wout_, Mask(mask_value)), USTR(spline_),
        Wide<const Dual2<float>>(wx_),
        // wknots_ is really a Wide<const Dual2<float>[]>,
        // but we are ignoring knot derivatives,
        // so just treat it as Wide<const float[]> which is binary compatible.
        Wide<const float[]>(wknots_, knot_arraylen), knot_count);
}


OSL_BATCHOP void
__OSL_MASKED_OP3(splineinverse, Wdf, Wdf,
                 df)(void* wout_, const char* spline_, void* wx_, void* wknots_,
                     int knot_count, int knot_arraylen, unsigned int mask_value)
{
    // Ignore knot derivatives
    __OSL_MASKED_OP3(splineinverse, Wdf, Wdf, f)
    (wout_, spline_, wx_, wknots_, knot_count, knot_arraylen, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(splineinverse, Wdf, df, Wdf)(void* wout_, const char* spline_,
                                              void* wx_, void* wknots_,
                                              int knot_count, int knot_arraylen,
                                              unsigned int mask_value)
{
    // Ignore knot derivatives
    splineinverse_evaluate_wide(
        Masked<Dual2<float>>(wout_, Mask(mask_value)), USTR(spline_),
        UniformAsWide<const Dual2<float>>(wx_),
        // wknots_ is really a Wide<const Dual2<float>[]>,
        // but we are ignoring knot derivatives,
        // so just treat it as Wide<const float[]> which is binary compatible.
        Wide<const float[]>(wknots_, knot_arraylen), knot_count);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(splineinverse, Wdf, f, Wdf)(void* wout_, const char* spline_,
                                             void* wx_, void* wknots_,
                                             int knot_count, int knot_arraylen,
                                             unsigned int mask_value)
{
    // Ignore knot derivs
    // treated as fff
    __OSL_MASKED_OP3(splineinverse, Wf, f, Wf)
    (wout_, spline_, wx_, wknots_, knot_count, knot_arraylen, mask_value);

    // Clear the ONLY the derivatives of a masked Dual2<float>,
    // leave the value alone.
    Mask mask(mask_value);
    MaskedDx<float> woutDx(wout_, mask);
    MaskedDy<float> woutDy(wout_, mask);
    assign_all(woutDx, 0.0f);
    assign_all(woutDy, 0.0f);
}

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
