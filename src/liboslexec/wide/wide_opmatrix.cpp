// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of matrix operations.
///
/////////////////////////////////////////////////////////////////////////
#include <OSL/oslconfig.h>

#include <OSL/batched_rendererservices.h>
#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

using BatchedRendererServices = OSL::BatchedRendererServices<__OSL_WIDTH>;

#include "define_opname_macros.h"

namespace {

// invoke is helper to ensure a functor is compiled inside a
// actual function call vs. inlining.
// Useful for keeping the slow path from fusing
// with a streamlined SIMD loop or influencing
// register usage
template<typename FunctorT>
OSL_NOINLINE void
invoke(FunctorT f);

template<typename FunctorT>
void
invoke(FunctorT f)
{
    f();
}


// helper to make a Block<ustringhash> from a pointer
#if !OSL_USTRINGREP_IS_HASH
inline void
block_ustringhash_from_ptr(Block<ustringhash>& b, const void* w_ptr)
{
    for (int i = 0; i < __OSL_WIDTH; ++i)
        b.set(i, reinterpret_cast<const ustring*>(w_ptr)[i]);
}
#endif


OSL_FORCEINLINE void
invert_wide_matrix(Masked<Matrix44> wresult, Wide<const Matrix44> wmatrix)
{
    if (wresult.mask().any_on()) {
        Block<int> notAffineBlock;
        Wide<int> wnotAffine(notAffineBlock);

        OSL_FORCEINLINE_BLOCK
        {
            OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
            for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
                Matrix44 m     = wmatrix[lane];
                bool is_affine = true;
                if (wresult.mask()[lane]) {
                    is_affine = test_if_affine(m);
                    if (OSL_UNLIKELY(is_affine)) {
                        Matrix44 r                = OSL::affineInverse(m);
                        wresult[ActiveLane(lane)] = r;
                    }
                }
                wnotAffine[lane]
                    = (!is_affine);  // false when lane is masked off
            }
        }

        if (testIfAnyLaneIsNonZero(wnotAffine)) {
            invoke([=]() -> void {
                for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
                    if (wnotAffine[lane]) {
                        OSL_DASSERT(wresult.mask().is_on(lane));
                        Matrix44 m                = wmatrix[lane];
                        Matrix44 invm             = OSL::nonAffineInverse(m);
                        wresult[ActiveLane(lane)] = invm;
                    }
                }
            });
        }
    }
}


Mask
default_get_matrix(BatchedRendererServices* bsr, BatchedShaderGlobals* bsg,
                   Masked<Matrix44> wresult, Wide<const ustringhash> wfrom,
                   Wide<const float> wtime)
{
    Mask ok(false);
    foreach_unique(wfrom, wresult.mask(),
                   [=, &ok](const ustringhash& from, Mask from_mask) {
                       // Reuse the uniform from implementation by restricting results to
                       // just the lanes with the same value of "from".
                       Masked<Matrix44> wsub_result(wresult.data(), from_mask);
                       Mask sub_ok = bsr->get_matrix(bsg, wsub_result, from,
                                                     wtime);
                       ok |= sub_ok;
                   });
    return ok;
}

// Avoid calling virtual functions and allow the default implementations
// to exist in target specific libraries.  We use a dispatch helper
// to call the virtual function ONLY if it is overridden, otherwise
// execute the ISA optimized default version built right here.
OSL_FORCEINLINE Mask
dispatch_get_matrix(BatchedRendererServices* bsr, BatchedShaderGlobals* bsg,
                    Masked<Matrix44> result, Wide<const ustringhash> from,
                    Wide<const float> time)
{
    if (bsr->is_overridden_get_matrix_WmWsWf()) {
        return bsr->get_matrix(bsg, result, from, time);
    } else {
        return default_get_matrix(bsr, bsg, result, from, time);
    }
}

Mask
default_get_inverse_matrix(BatchedRendererServices* bsr,
                           BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                           Wide<const TransformationPtr> xform,
                           Wide<const float> time)
{
    OSL_FORCEINLINE_BLOCK
    {
        Block<Matrix44> wmatrix;
        Mask succeeded
            = bsr->get_matrix(bsg, Masked<Matrix44>(wmatrix, result.mask()),
                              xform, time);
        invert_wide_matrix(result & succeeded, wmatrix);
        return succeeded;
    }
}

OSL_FORCEINLINE Mask
dispatch_get_inverse_matrix(BatchedRendererServices* bsr,
                            BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                            Wide<const TransformationPtr> xform,
                            Wide<const float> time)
{
    if (bsr->is_overridden_get_inverse_matrix_WmWxWf()) {
        return bsr->get_inverse_matrix(bsg, result, xform, time);
    } else {
        return default_get_inverse_matrix(bsr, bsg, result, xform, time);
    }
}


Mask
default_get_inverse_matrix(BatchedRendererServices* bsr,
                           BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                           ustringhash to, Wide<const float> time)
{
    OSL_FORCEINLINE_BLOCK
    {
        Block<Matrix44> wmatrix;
        Mask succeeded
            = bsr->get_matrix(bsg, Masked<Matrix44>(wmatrix, result.mask()), to,
                              time);
        invert_wide_matrix(result & succeeded, wmatrix);
        return succeeded;
    }
}

OSL_FORCEINLINE Mask
dispatch_get_inverse_matrix(BatchedRendererServices* bsr,
                            BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                            ustringhash to, Wide<const float> time)
{
    if (bsr->is_overridden_get_inverse_matrix_WmsWf()) {
        return bsr->get_inverse_matrix(bsg, result, to, time);
    } else {
        return default_get_inverse_matrix(bsr, bsg, result, to, time);
    }
}


Mask
default_get_inverse_matrix(BatchedRendererServices* bsr,
                           BatchedShaderGlobals* bsg, Masked<Matrix44> wresult,
                           Wide<const ustringhash> wto, Wide<const float> wtime)
{
    if (bsr->is_overridden_get_inverse_matrix_WmsWf()) {
        Mask ok(false);
        foreach_unique(wto, wresult.mask(),
                       [=, &ok](const ustringhash& to, Mask from_mask) {
                           // Reuse the uniform from implementation by restricting results to
                           // just the lanes with the same value of "from".
                           Masked<Matrix44> wsub_result(wresult.data(),
                                                        from_mask);
                           Mask sub_ok = bsr->get_inverse_matrix(bsg,
                                                                 wsub_result,
                                                                 to, wtime);
                           ok |= sub_ok;
                       });
        return ok;
    } else {
        OSL_FORCEINLINE_BLOCK
        {
            Block<Matrix44> wmatrix;
            Mask succeeded = dispatch_get_matrix(bsr, bsg, wresult, wto, wtime);
            invert_wide_matrix(wresult & succeeded, wmatrix);
            return succeeded;
        }
    }
}

OSL_FORCEINLINE Mask
dispatch_get_inverse_matrix(BatchedRendererServices* bsr,
                            BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                            Wide<const ustringhash> to, Wide<const float> time)
{
    if (bsr->is_overridden_get_inverse_matrix_WmWsWf()) {
        return bsr->get_inverse_matrix(bsg, result, to, time);
    } else {
        return default_get_inverse_matrix(bsr, bsg, result, to, time);
    }
}

}  // namespace

OSL_BATCHOP void
__OSL_OP3(mul, Wm, Wm, Wf)(void* wr_, void* wa_, void* wb_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wa(wa_);
        Wide<const float> wb(wb_);
        Wide<Matrix44> wr(wr_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 a = wa[lane];
            float b    = wb[lane];
            Matrix44 r = a * b;
            wr[lane]   = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(mul, Wm, Wm, Wf)(void* wr_, void* wa_, void* wb_,
                                  unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wa(wa_);
        Wide<const float> wb(wb_);
        Masked<Matrix44> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 a = wa[lane];
            float b    = wb[lane];
            Matrix44 r = a * b;
            wr[lane]   = r;
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(mul, Wm, Wm, Wm)(void* wr_, void* wa_, void* wb_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wa(wa_);
        Wide<const Matrix44> wb(wb_);
        Wide<Matrix44> wr(wr_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 a = wa[lane];
            Matrix44 b = wb[lane];
            // Need inlinable version for vectorization
            // Matrix44 r = a * b;
            // TODO: replace Matrix * Matrix implementation
            // in IMATH with inlinable equivalent of multiplyMatrixByMatrix
            wr[lane] = multiplyMatrixByMatrix(a, b);
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(mul, Wm, Wm, Wm)(void* wr_, void* wa_, void* wb_,
                                  unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wa(wa_);
        Wide<const Matrix44> wb(wb_);
        Masked<Matrix44> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 a = wa[lane];
            Matrix44 b = wb[lane];
            // Need inlinable version for vectorization
            // Matrix44 r = a * b;
            wr[lane] = multiplyMatrixByMatrix(a, b);
        }
    }
}


OSL_BATCHOP void
__OSL_MASKED_OP3(div, Wm, Wm, Wm)(void* wr_, void* wa_, void* wb_,
                                  unsigned int mask_value)
{
    Wide<const Matrix44> wa(wa_);
    Wide<const Matrix44> wb(wb_);
    Masked<Matrix44> wresult(wr_, Mask(mask_value));

    Block<int> notAffineBlock;
    Wide<int> wnotAffine(notAffineBlock);

    // Rather than calling b.inverse() which has to handle affine and
    // non-affine matrices, we will test if b is affine and only
    // vectorize the fast path and create a test to skip the
    // slow path for non-affine inverses and avoiding attempting to
    // vectorize the slow path.
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 a     = wa[lane];
            Matrix44 b     = wb[lane];
            bool is_affine = true;
            if (wresult.mask()[lane]) {
                is_affine = test_if_affine(b);
                if (OSL_UNLIKELY(is_affine)) {
                    wresult[ActiveLane(lane)]
                        = multiplyMatrixByMatrix(a, OSL::affineInverse(b));
                }
            }
            wnotAffine[lane] = (!is_affine);  // false when lane is masked off
        }
    }

    if (testIfAnyLaneIsNonZero(wnotAffine)) {
        invoke([=]() -> void {
            // DO NOT VECTORIZE the slow path
            for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
                if (wnotAffine[lane]) {
                    OSL_DASSERT(wresult.mask().is_on(lane));
                    Matrix44 a                = wa[lane];
                    Matrix44 b                = wb[lane];
                    Matrix44 r                = a * OSL::nonAffineInverse(b);
                    wresult[ActiveLane(lane)] = r;
                }
            }
        });
    }
}



OSL_BATCHOP void
__OSL_OP3(div, Wm, Wm, Wf)(void* wr_, void* wa_, void* wb_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wa(wa_);
        Wide<const float> wb(wb_);
        Wide<Matrix44> wr(wr_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 a = wa[lane];
            float b    = wb[lane];
            Matrix44 r = a * (1.0f / b);
            wr[lane]   = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(div, Wm, Wm, Wf)(void* wr_, void* wa_, void* wb_,
                                  unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wa(wa_);
        Wide<const float> wb(wb_);
        Masked<Matrix44> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 a = wa[lane];
            float b    = wb[lane];
            if (wr.mask()[lane]) {
                Matrix44 r           = a * (1.0f / b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}


OSL_BATCHOP void
__OSL_MASKED_OP3(div, Wm, Wf, Wm)(void* wr_, void* wa_, void* wb_,
                                  unsigned int mask_value)
{
    Wide<const float> wa(wa_);
    Wide<const Matrix44> wb(wb_);
    Masked<Matrix44> wresult(wr_, Mask(mask_value));

    Block<int> notAffineBlock;
    Wide<int> wnotAffine(notAffineBlock);

    // Rather than calling b.inverse() which has to handle affine and
    // non-affine matrices, we will test if b is affine and only
    // vectorize the fast path and create a test to skip the
    // slow path for non-affine inverses and avoiding attempting to
    // vectorize the slow path.
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const float a    = wa[lane];
            const Matrix44 b = wb[lane];
            bool is_affine   = true;
            if (wresult.mask()[lane]) {
                is_affine = test_if_affine(b);
                if (OSL_UNLIKELY(is_affine)) {
                    Matrix44 r = a * OSL::affineInverse(b);

                    wresult[ActiveLane(lane)] = r;
                }
            }
            wnotAffine[lane] = (!is_affine);  // false when lane is masked off
        }
    }

    if (testIfAnyLaneIsNonZero(wnotAffine)) {
        invoke([=]() -> void {
            // DO NOT VECTORIZE the slow path
            for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
                if (wnotAffine[lane]) {
                    OSL_DASSERT(wresult.mask().is_on(lane));
                    float a                   = wa[lane];
                    Matrix44 b                = wb[lane];
                    Matrix44 r                = a * OSL::nonAffineInverse(b);
                    wresult[ActiveLane(lane)] = r;
                }
            }
        });
    }
}



OSL_BATCHOP void
__OSL_OP2(transpose, Wm, Wm)(void* wr_, void* wm_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wm(wm_);
        Wide<Matrix44> wr(wr_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 m = wm[lane];
            // Call inlineable transposed
            //Matrix44 r = m.transposed();
            // TODO: replace Matrix::transposed implementation
            // in IMATH with equivalent of inlinedTransposed
            Matrix44 r = inlinedTransposed(m);
            wr[lane]   = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(transpose, Wm, Wm)(void* wr_, void* wm_,
                                    unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wm(wm_);
        Masked<Matrix44> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 m = wm[lane];
            // Call inlineable transposed
            //Matrix44 r = m.transposed();
            Matrix44 r = inlinedTransposed(m);
            wr[lane]   = r;
        }
    }
}

namespace {
OSL_NOINLINE void
makeIdentity(Masked<Matrix44> wrm)
{
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 ident(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
            wrm[lane] = ident;
        }
    }
}

OSL_FORCEINLINE Mask
impl_get_uniform_from_matrix_masked(void* bsg_, Masked<Matrix44> wrm,
                                    const char* from)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;

    if (USTR(from) == Strings::common
        || USTR(from) == ctx->shadingsys().commonspace_synonym()) {
        makeIdentity(wrm);

        return wrm.mask();
    }

    if (USTR(from) == Strings::shader) {
        ctx->batched<__OSL_WIDTH>().renderer()->get_matrix(
            bsg, wrm, bsg->varying.shader2common, bsg->varying.time);
        // NOTE: matching scalar version of code which ignores the renderservices return value
        return wrm.mask();
    }
    if (USTR(from) == Strings::object) {
        ctx->batched<__OSL_WIDTH>().renderer()->get_matrix(
            bsg, wrm, bsg->varying.object2common, bsg->varying.time);
        // NOTE: matching scalar version of code which ignores the renderservices return value
        return wrm.mask();
    }

    Mask succeeded = ctx->batched<__OSL_WIDTH>().renderer()->get_matrix(
        bsg, wrm, USTR(from), bsg->varying.time);
    auto failedResults = wrm & succeeded.invert();
    if (failedResults.mask().any_on()) {
        makeIdentity(failedResults);
        ShadingContext* ctx = bsg->uniform.context;
        if (ctx->shadingsys().unknown_coordsys_error()) {
            ctx->errorfmt("Unknown transformation \"{}\"", from);
        }
    }
    return succeeded;
}

OSL_FORCEINLINE Mask
impl_get_uniform_to_inverse_matrix_masked(void* bsg_, Masked<Matrix44> wrm,
                                          const char* to)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;

    if (USTR(to) == Strings::common
        || USTR(to) == ctx->shadingsys().commonspace_synonym()) {
        makeIdentity(wrm);
        return wrm.mask();
    }
    if (USTR(to) == Strings::shader) {
        dispatch_get_inverse_matrix(ctx->batched<__OSL_WIDTH>().renderer(), bsg,
                                    wrm, bsg->varying.shader2common,
                                    bsg->varying.time);
        // NOTE: matching scalar version of code which ignores the renderservices return value
        return wrm.mask();
    }
    if (USTR(to) == Strings::object) {
        dispatch_get_inverse_matrix(ctx->batched<__OSL_WIDTH>().renderer(), bsg,
                                    wrm, bsg->varying.object2common,
                                    bsg->varying.time);
        // NOTE: matching scalar version of code which ignores the renderservices return value
        return wrm.mask();
    }

    // Based on the 1 function that calls this function
    // the results of the failed data lanes will get overwritten
    // so no need to make sure that the values are valid (assuming FP exceptions are disabled)
    Mask succeeded
        = dispatch_get_inverse_matrix(ctx->batched<__OSL_WIDTH>().renderer(),
                                      bsg, wrm, USTR(to), bsg->varying.time);

    auto failedResults = wrm & succeeded.invert();
    if (failedResults.mask().any_on()) {
        makeIdentity(failedResults);
        if (ctx->shadingsys().unknown_coordsys_error()) {
            ctx->errorfmt("Unknown transformation \"{}\"", to);
        }
    }
    return succeeded;
}

template<typename ResultAccessorT, typename FromAccessorT, typename ToAccessorT>
OSL_FORCEINLINE void
impl_wide_mat_multiply(ResultAccessorT wresult, FromAccessorT wfrom,
                       ToAccessorT wto)
{
    OSL_FORCEINLINE_BLOCK
    {
        // No savings from using a WeakMask
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 mat_From = wfrom[lane];
            Matrix44 mat_To   = wto[lane];

            // Need to call inlinable version
            //Matrix44 result = mat_From * mat_To;
            wresult[lane] = multiplyMatrixByMatrix(mat_From, mat_To);
        }
    }
}

OSL_FORCEINLINE Mask
impl_get_varying_from_matrix_batched(BatchedShaderGlobals* bsg,
                                     ShadingContext* ctx,
                                     Wide<const ustringhash> wFrom,
                                     Masked<Matrix44> wMfrom)
{
    // Deal with a varying 'from' space
    ustring commonspace_synonym = ctx->shadingsys().commonspace_synonym();

    // Use int instead of Mask<> to allow reduction clause in openmp simd declaration
    int common_space_bits { 0 };
    int shader_space_bits { 0 };
    int object_space_bits { 0 };
    int named_space_bits { 0 };

    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH)
                           reduction(|
                                     : common_space_bits, shader_space_bits,
                                       object_space_bits, named_space_bits))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            ustringhash from = wFrom[lane];
            if (wMfrom.mask()[lane]) {
                if (from == Strings::common || from == commonspace_synonym) {
                    // inline of Mask::set_on(lane)
                    common_space_bits |= 1 << lane;
                } else if (from == Strings::shader) {
                    // inline of Mask::set_on(lane)
                    shader_space_bits |= 1 << lane;
                } else if (from == Strings::object) {
                    // inline of Mask::set_on(lane)
                    object_space_bits |= 1 << lane;
                } else {
                    // inline of Mask::set_on(lane)
                    named_space_bits |= 1 << lane;
                }
            }
        }
    }
    Mask common_space_mask(common_space_bits);
    Mask shader_space_mask(shader_space_bits);
    Mask object_space_mask(object_space_bits);
    Mask named_space_mask(named_space_bits);

    if (common_space_mask.any_on()) {
        Masked<Matrix44> mfrom(wMfrom.data(), common_space_mask);
        makeIdentity(mfrom);
    }
    const auto& sgbv = bsg->varying;
    if (shader_space_mask.any_on()) {
        Masked<Matrix44> mfrom(wMfrom.data(), shader_space_mask);
        ctx->batched<__OSL_WIDTH>().renderer()->get_matrix(bsg, mfrom,
                                                           sgbv.shader2common,
                                                           sgbv.time);
        // NOTE: matching scalar version of code which ignores the renderservices return value
    }
    if (object_space_mask.any_on()) {
        Masked<Matrix44> mfrom(wMfrom.data(), object_space_mask);
        ctx->batched<__OSL_WIDTH>().renderer()->get_matrix(bsg, mfrom,
                                                           sgbv.object2common,
                                                           sgbv.time);
        // NOTE: matching scalar version of code which ignores the renderservices return value
    }
    // Only named lookups can fail, so we can just subtract those lanes
    Mask succeeded(wMfrom.mask());
    if (named_space_mask.any_on()) {
        Masked<Matrix44> mfrom(wMfrom.data(), named_space_mask);

        Mask success = ctx->batched<__OSL_WIDTH>().renderer()->get_matrix(
            bsg, mfrom, wFrom, sgbv.time);

        Mask failedLanes = success.invert() & named_space_mask;
        if (failedLanes.any_on()) {
            Masked<Matrix44> mto_failed(wMfrom.data(), failedLanes);
            makeIdentity(mto_failed);
            if (ctx->shadingsys().unknown_coordsys_error()) {
                for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
                    if (failedLanes[lane]) {
                        ustringhash from = wFrom[lane];
                        ctx->batched<__OSL_WIDTH>().errorfmt(
                            Mask(Lane(lane)), "Unknown transformation \"{}\"",
                            ustring(from));
                    }
                }
            }

            // Remove any failed lanes from the success mask
            succeeded &= ~failedLanes;
        }
    }
    return succeeded;
}
}  // namespace

OSL_BATCHOP void
__OSL_MASKED_OP2(prepend_matrix_from, Wm, s)(void* bsg_, void* wr,
                                             const char* from,
                                             unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    Block<Matrix44> wMfrom;
    Masked<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    /*Mask succeeded =*/
    impl_get_uniform_from_matrix_masked(bsg, from_matrix, from);

    Masked<Matrix44> wrm(wr, Mask(mask_value));

    impl_wide_mat_multiply(wrm, from_matrix, wrm);
}



OSL_BATCHOP void
__OSL_MASKED_OP2(prepend_matrix_from, Wm, Ws)(void* bsg_, void* wr,
                                              void* w_from_name,
                                              unsigned int mask_value)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;

#if OSL_USTRINGREP_IS_HASH
    Wide<const ustringhash> wFromName(w_from_name);
#else
    Block<ustringhash> bwFromName;
    block_ustringhash_from_ptr(bwFromName, w_from_name);
    Wide<const ustringhash> wFromName(bwFromName);
#endif

    Block<Matrix44> wMfrom;
    Masked<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    /*Mask succeeded =*/
    impl_get_varying_from_matrix_batched(bsg, ctx, wFromName, from_matrix);

    Masked<Matrix44> wrm(wr, Mask(mask_value));

    impl_wide_mat_multiply(wrm, from_matrix, wrm);
}


namespace {
OSL_FORCEINLINE Mask
impl_get_varying_to_matrix_masked(BatchedShaderGlobals* bsg,
                                  ShadingContext* ctx,
                                  Wide<const ustringhash> wTo,
                                  Masked<Matrix44> wMto)
{
    // Deal with a varying 'to' space
    ustring commonspace_synonym = ctx->shadingsys().commonspace_synonym();

    // Use int instead of Mask<> to allow reduction clause in openmp simd declaration
    int common_space_bits { 0 };
    int shader_space_bits { 0 };
    int object_space_bits { 0 };
    int named_space_bits { 0 };

    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH)
                           reduction(|
                                     : common_space_bits, shader_space_bits,
                                       object_space_bits, named_space_bits))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            ustringhash to = wTo[lane];
            if (wMto.mask()[lane]) {
                if (to == Strings::common || to == commonspace_synonym) {
                    // inline of Mask::set_on(lane)
                    common_space_bits |= 1 << lane;
                } else if (to == Strings::shader) {
                    // inline of Mask::set_on(lane)
                    shader_space_bits |= 1 << lane;
                } else if (to == Strings::object) {
                    // inline of Mask::set_on(lane)
                    object_space_bits |= 1 << lane;
                } else {
                    // inline of Mask::set_on(lane)
                    named_space_bits |= 1 << lane;
                }
            }
        }
    }
    Mask common_space_mask(common_space_bits);
    Mask shader_space_mask(shader_space_bits);
    Mask object_space_mask(object_space_bits);
    Mask named_space_mask(named_space_bits);

    if (common_space_mask.any_on()) {
        Masked<Matrix44> mto(wMto.data(), common_space_mask);
        makeIdentity(mto);
    }
    const auto& sgbv = bsg->varying;
    if (shader_space_mask.any_on()) {
        Masked<Matrix44> mto(wMto.data(), shader_space_mask);
        dispatch_get_inverse_matrix(ctx->batched<__OSL_WIDTH>().renderer(), bsg,
                                    mto, sgbv.shader2common, sgbv.time);
        // NOTE: matching scalar version of code which ignores the renderservices return value
    }
    if (object_space_mask.any_on()) {
        Masked<Matrix44> mto(wMto.data(), object_space_mask);
        dispatch_get_inverse_matrix(ctx->batched<__OSL_WIDTH>().renderer(), bsg,
                                    mto, sgbv.object2common, sgbv.time);
        // NOTE: matching scalar version of code which ignores the renderservices return value
    }
    // Only named lookups can fail, so we can just subtract those lanes
    Mask succeeded(wMto.mask());
    if (named_space_mask.any_on()) {
        Masked<Matrix44> mto(wMto.data(), named_space_mask);

        Mask success = dispatch_get_inverse_matrix(
            ctx->batched<__OSL_WIDTH>().renderer(), bsg, mto, wTo, sgbv.time);

        Mask failedLanes = success.invert() & named_space_mask;
        if (failedLanes.any_on()) {
            Masked<Matrix44> mto(wMto.data(), failedLanes);
            makeIdentity(mto);
            if (ctx->shadingsys().unknown_coordsys_error()) {
                for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
                    if (failedLanes[lane]) {
                        ustringhash to = wTo[lane];
                        ctx->batched<__OSL_WIDTH>().errorfmt(
                            Mask(Lane(lane)), "Unknown transformation \"{}\"",
                            ustring(to));
                    }
                }
            }

            // Remove any failed lanes from the success mask
            succeeded &= ~failedLanes;
        }
    }
    return succeeded;
}


OSL_FORCEINLINE Mask
impl_get_uniform_from_to_matrix_masked(BatchedShaderGlobals* bsg,
                                       Masked<Matrix44> wrm, const char* from,
                                       const char* to)
{
    Block<Matrix44> wMfrom, wMto;
    Masked<Matrix44> from_matrix(wMfrom, wrm.mask());
    Mask succeeded = impl_get_uniform_from_matrix_masked(bsg, from_matrix,
                                                         from);

    // NOTE: even if we failed to get a from matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    Masked<Matrix44> to_matrix(wMto, wrm.mask());
    succeeded &= impl_get_uniform_to_inverse_matrix_masked(bsg, to_matrix, to);

    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded;
}
}  // namespace

OSL_BATCHOP int
__OSL_MASKED_OP3(get_from_to_matrix, Wm, s, s)(void* bsg_, void* wr,
                                               const char* from, const char* to,
                                               unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    Masked<Matrix44> wrm(wr, Mask(mask_value));
    return impl_get_uniform_from_to_matrix_masked(bsg, wrm, from, to).value();
}


OSL_BATCHOP int
__OSL_MASKED_OP3(get_from_to_matrix, Wm, s,
                 Ws)(void* bsg_, void* wr, const char* from, void* w_to_ptr,
                     unsigned int mask_value)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;
    Block<Matrix44> wMfrom;
    Masked<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    Mask succeeded = impl_get_uniform_from_matrix_masked(bsg, from_matrix,
                                                         from);

#if OSL_USTRINGREP_IS_HASH
    Wide<const ustringhash> wToSpace(w_to_ptr);
#else
    Block<ustringhash> bwToSpace;
    block_ustringhash_from_ptr(bwToSpace, w_to_ptr);
    Wide<const ustringhash> wToSpace(bwToSpace);
#endif
    Block<Matrix44> wMto;
    // NOTE: even if we failed to get a from matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    Masked<Matrix44> to_matrix(wMto, Mask(mask_value));
    succeeded &= impl_get_varying_to_matrix_masked(bsg, ctx, wToSpace,
                                                   to_matrix);

    Masked<Matrix44> wrm(wr, Mask(mask_value));
    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded.value();
}


OSL_BATCHOP int
__OSL_MASKED_OP3(get_from_to_matrix, Wm, Ws,
                 s)(void* bsg_, void* wr, void* w_from_ptr, const char* to,
                    unsigned int mask_value)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;

#if OSL_USTRINGREP_IS_HASH
    Wide<const ustringhash> wFromName(w_from_ptr);
#else
    Block<ustringhash> bwFromName;
    block_ustringhash_from_ptr(bwFromName, w_from_ptr);
    Wide<const ustringhash> wFromName(bwFromName);
#endif

    Block<Matrix44> wMto;
    Masked<Matrix44> to_matrix(wMto, Mask(mask_value));
    Mask succeeded = impl_get_uniform_to_inverse_matrix_masked(bsg, to_matrix,
                                                               to);

    Block<Matrix44> wMfrom;
    // NOTE: even if we failed to get a to matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    Masked<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    succeeded &= impl_get_varying_from_matrix_batched(bsg, ctx, wFromName,
                                                      from_matrix);

    Masked<Matrix44> wrm(wr, Mask(mask_value));
    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded.value();
}


OSL_BATCHOP int
__OSL_MASKED_OP3(get_from_to_matrix, Wm, Ws,
                 Ws)(void* bsg_, void* wr, void* w_from_ptr, void* w_to_ptr,
                     unsigned int mask_value)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;

#if OSL_USTRINGREP_IS_HASH
    Wide<const ustringhash> wFromName(w_from_ptr);
#else
    Block<ustringhash> bwFromName;
    block_ustringhash_from_ptr(bwFromName, w_from_ptr);
    Wide<const ustringhash> wFromName(bwFromName);
#endif

    Block<Matrix44> wMfrom;
    Masked<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    Mask succeeded = impl_get_varying_from_matrix_batched(bsg, ctx, wFromName,
                                                          from_matrix);

#if OSL_USTRINGREP_IS_HASH
    Wide<const ustringhash> wToSpace(w_to_ptr);
#else
    Block<ustringhash> bwToSpace;
    block_ustringhash_from_ptr(bwToSpace, w_to_ptr);
    Wide<const ustringhash> wToSpace(bwToSpace);
#endif
    Block<Matrix44> wMto;
    // NOTE: even if we failed to get a from matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    Masked<Matrix44> to_matrix(wMto, Mask(mask_value));
    succeeded &= impl_get_varying_to_matrix_masked(bsg, ctx, wToSpace,
                                                   to_matrix);

    Masked<Matrix44> wrm(wr, Mask(mask_value));
    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded.value();
}


// NOTE:  For batched transforms, a different dispatch approach is used.
// Instead of calling a single transform_triple function with lots of
// conditionals to select/dispatch the correct code, a 2 steps are taken.
// First call an explicitly named function (osl_build_transform_matrix_??_masked)
// is called that represents the uniformity for the different from & to spaces
// is called building a transform matrix.
// Second call an explicitly named function (osl_transform_[point|vector|normal]_??_masked)
// is called that represents the uniformity and data types of the src and destination triples.
// Also zeroing of derivatives is left to the code generator.

OSL_BATCHOP int
__OSL_MASKED_OP3(build_transform_matrix, Wm, s,
                 s)(void* bsg_, void* WM_, ustring_pod from_, ustring_pod to_,
                    unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    Mask mask(mask_value);
    Masked<Matrix44> mm(WM_, mask);
    ShadingContext* ctx = bsg->uniform.context;

    ustringrep from = USTR(from_);
    ustringrep to   = USTR(to_);

    Mask succeeded;
    // Avoid matrix concatenation if possible by detecting when the
    // adjacent matrix would be identity
    // We don't expect both from and to == common, so we are not
    // optimizing for it
    if (from == Strings::common
        || from == ctx->shadingsys().commonspace_synonym()) {
        succeeded = impl_get_uniform_to_inverse_matrix_masked(bsg, mm,
                                                              to.c_str());
    } else if (to == Strings::common
               || to == ctx->shadingsys().commonspace_synonym()) {
        succeeded = impl_get_uniform_from_matrix_masked(bsg, mm, from.c_str());
    } else {
        succeeded = impl_get_uniform_from_to_matrix_masked(bsg, mm,
                                                           from.c_str(),
                                                           to.c_str());
    }
    return succeeded.value();
}



OSL_BATCHOP int
__OSL_MASKED_OP3(build_transform_matrix, Wm, Ws,
                 s)(void* bsg_, void* WM_, void* wfrom_, ustring_pod to_,
                    unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    Mask mask(mask_value);
    Masked<Matrix44> wrm(WM_, mask);

#if OSL_USTRINGREP_IS_HASH
    Wide<const ustringhash> wfrom_space(wfrom_);
#else
    Block<ustringhash> bwfrom_space;
    block_ustringhash_from_ptr(bwfrom_space, wfrom_);
    Wide<const ustringhash> wfrom_space(bwfrom_space);
#endif

    ustringrep to_space = USTR(to_);

    Block<Matrix44> wMfrom, wMto;
    Masked<Matrix44> from_matrix(wMfrom, wrm.mask());
    ShadingContext* ctx = bsg->uniform.context;

    Mask succeeded = impl_get_varying_from_matrix_batched(bsg, ctx, wfrom_space,
                                                          from_matrix);
    Masked<Matrix44> to_matrix(wMto, wrm.mask() & succeeded);
    succeeded &= impl_get_uniform_to_inverse_matrix_masked(bsg, to_matrix,
                                                           to_space.c_str());

    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded.value();
}



OSL_BATCHOP int
__OSL_MASKED_OP3(build_transform_matrix, Wm, s,
                 Ws)(void* bsg_, void* WM_, ustring_pod from_, void* wto_,
                     unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    Mask mask(mask_value);
    Masked<Matrix44> wrm(WM_, mask);

    ustringrep from = USTR(from_);
#if OSL_USTRINGREP_IS_HASH
    Wide<const ustringhash> wto_space(wto_);
#else
    Block<ustringhash> bwto_space;
    block_ustringhash_from_ptr(bwto_space, wto_);
    Wide<const ustringhash> wto_space(bwto_space);
#endif

    Block<Matrix44> wMfrom, wMto;
    Masked<Matrix44> from_matrix(wMfrom, wrm.mask());
    ShadingContext* ctx = bsg->uniform.context;

    Mask succeeded = impl_get_uniform_from_matrix_masked(bsg, from_matrix,
                                                         from.c_str());
    Masked<Matrix44> to_matrix(wMto, wrm.mask() & succeeded);
    succeeded &= impl_get_varying_to_matrix_masked(bsg, ctx, wto_space,
                                                   to_matrix);

    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);

    return succeeded.value();
}



OSL_BATCHOP int
__OSL_MASKED_OP3(build_transform_matrix, Wm, Ws, Ws)(void* bsg_, void* WM_,
                                                     void* wfrom_, void* wto_,
                                                     unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    Mask mask(mask_value);
    Masked<Matrix44> wrm(WM_, mask);

#if OSL_USTRINGREP_IS_HASH
    Wide<const ustringhash> wfrom_space(wfrom_);
    Wide<const ustringhash> wto_space(wto_);
#else
    Block<ustringhash> bwfrom_space;
    block_ustringhash_from_ptr(bwfrom_space, wfrom_);
    Wide<const ustringhash> wfrom_space(bwfrom_space);
    Block<ustringhash> bwto_space;
    block_ustringhash_from_ptr(bwto_space, wto_);
    Wide<const ustringhash> wto_space(bwto_space);
#endif

    Block<Matrix44> wMfrom, wMto;
    Masked<Matrix44> from_matrix(wMfrom, wrm.mask());
    ShadingContext* ctx = bsg->uniform.context;

    Mask succeeded = impl_get_varying_from_matrix_batched(bsg, ctx, wfrom_space,
                                                          from_matrix);
    Masked<Matrix44> to_matrix(wMto, wrm.mask() & succeeded);
    succeeded &= impl_get_varying_to_matrix_masked(bsg, ctx, wto_space,
                                                   to_matrix);

    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);

    return succeeded.value();
}

namespace {
template<typename InputAccessorT>
OSL_FORCEINLINE void
impl_copy_untransformed_lanes(InputAccessorT inVec, void* Pout, Mask succeeded,
                              Mask op_mask)
{
    typedef typename InputAccessorT::NonConstValueType DataType;
    // if Pin != Pout, we still need to copy inactive data over to Pout
    // Handle cleaning up any data lanes that did not succeed
    if (((void*)&inVec.data() != Pout)) {
        // For any lanes we failed to get a matrix for
        // just copy the input to the output values
        // NOTE:  As we only only want to copy lanes that failed,
        // we will invert our success mask
        Mask failed = succeeded.invert() & op_mask;
        if (OSL_UNLIKELY(failed.any_on())) {
            OSL_FORCEINLINE_BLOCK
            {
                Masked<DataType> failedOutVec(Pout, failed);
                OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
                for (int i = 0; i < __OSL_WIDTH; ++i) {
                    failedOutVec[i] = inVec[i];
                }
            }
        }
    }
}

template<typename InputAccessorT, typename MatrixAccessorT>
OSL_FORCEINLINE void
impl_transform_point_masked(void* Pin, void* Pout, void* transform,
                            unsigned int mask_transform,
                            unsigned int mask_value)
{
    typedef typename InputAccessorT::NonConstValueType DataType;

    // ignore derivs because output doesn't need it
    OSL_FORCEINLINE_BLOCK
    {
        Mask mask(mask_value);
        Mask succeeded(mask_transform);

        InputAccessorT inPoints(Pin);
        // only operate on active lanes
        Mask activeMask = mask & succeeded;

        Masked<DataType> wresult(Pout, activeMask);
        MatrixAccessorT wM(transform);

        // Transform with Vector semantics
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Matrix44 m = wM[lane];
            const DataType v = inPoints[lane];

            if (wresult.mask()[lane]) {
                DataType r;

                // Do to illegal aliasing in OpenEXR version
                // we call our own flavor without aliasing
                robust_multVecMatrix(m, v, r);

                wresult[ActiveLane(lane)] = r;
            }
        }

        impl_copy_untransformed_lanes(inPoints, Pout, succeeded, mask);
    }
}
}  // namespace

OSL_BATCHOP void
__OSL_MASKED_OP3(transform_point, v, Wv,
                 m)(void* Pin, void* Pout, void* transform,
                    unsigned int mask_transform, unsigned int mask_value)
{
    // TODO: see if we can get gen_transform to call the vvm version then do a masked broadcast
    impl_transform_point_masked<UniformAsWide<const Vec3>,
                                UniformAsWide<const Matrix44>>(Pin, Pout,
                                                               transform,
                                                               mask_transform,
                                                               mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_point, v, Wv,
                 Wm)(void* Pin, void* Pout, void* transform,
                     unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_point_masked<UniformAsWide<const Vec3>, Wide<const Matrix44>>(
        Pin, Pout, transform, mask_transform, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_point, Wv, Wv,
                 Wm)(void* Pin, void* Pout, void* transform,
                     unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_point_masked<Wide<const Vec3>, Wide<const Matrix44>>(
        Pin, Pout, transform, mask_transform, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_point, Wdv, Wdv,
                 Wm)(void* Pin, void* Pout, void* transform,
                     unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_point_masked<Wide<const Dual2<Vec3>>, Wide<const Matrix44>>(
        Pin, Pout, transform, mask_transform, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_point, Wv, Wv,
                 m)(void* Pin, void* Pout, void* transform,
                    unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_point_masked<Wide<const Vec3>, UniformAsWide<const Matrix44>>(
        Pin, Pout, transform, mask_transform, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_point, Wdv, Wdv,
                 m)(void* Pin, void* Pout, void* transform,
                    unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_point_masked<Wide<const Dual2<Vec3>>,
                                UniformAsWide<const Matrix44>>(Pin, Pout,
                                                               transform,
                                                               mask_transform,
                                                               mask_value);
}

namespace {
template<typename InputAccessorT, typename MatrixAccessorT>
OSL_FORCEINLINE void
impl_transform_vector_masked(void* Pin, void* Pout, void* transform,
                             unsigned int mask_transform,
                             unsigned int mask_value)
{
    typedef typename InputAccessorT::NonConstValueType DataType;

    // ignore derivs because output doesn't need it
    OSL_FORCEINLINE_BLOCK
    {
        Mask mask(mask_value);
        Mask succeeded(mask_transform);

        InputAccessorT inPoints(Pin);
        // only operate on active lanes
        Mask activeMask = mask & succeeded;

        Masked<DataType> wresult(Pout, activeMask);
        MatrixAccessorT wM(transform);

        // Transform with Vector semantics
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 m = wM[lane];
            DataType v = inPoints[lane];
            if (wresult.mask()[lane]) {
                // Do to illegal aliasing in OpenEXR version
                // we call our own flavor without aliasing
                //M.multDirMatrix (v, VEC(result));
                // TODO: update multDirMatrix to equivalent of multiplyDirByMatrix
                wresult[ActiveLane(lane)] = multiplyDirByMatrix(m, v);
            }
        }

        impl_copy_untransformed_lanes(inPoints, Pout, succeeded, mask);
    }
}
}  // namespace

OSL_BATCHOP void
__OSL_MASKED_OP3(transform_vector, v, Wv,
                 m)(void* Pin, void* Pout, void* transform,
                    unsigned int mask_transform, unsigned int mask_value)
{
    // TODO: see if we can get gen_transform to call the vvm version then do a masked broadcast
    impl_transform_vector_masked<UniformAsWide<const Vec3>,
                                 UniformAsWide<const Matrix44>>(Pin, Pout,
                                                                transform,
                                                                mask_transform,
                                                                mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_vector, v, Wv,
                 Wm)(void* Pin, void* Pout, void* transform,
                     unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_vector_masked<UniformAsWide<const Vec3>,
                                 Wide<const Matrix44>>(Pin, Pout, transform,
                                                       mask_transform,
                                                       mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_vector, Wv, Wv,
                 Wm)(void* Pin, void* Pout, void* transform,
                     unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_vector_masked<Wide<const Vec3>, Wide<const Matrix44>>(
        Pin, Pout, transform, mask_transform, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_vector, Wdv, Wdv,
                 Wm)(void* Pin, void* Pout, void* transform,
                     unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_vector_masked<Wide<const Dual2<Vec3>>, Wide<const Matrix44>>(
        Pin, Pout, transform, mask_transform, mask_value);
}


OSL_BATCHOP void
__OSL_MASKED_OP3(transform_vector, Wv, Wv,
                 m)(void* Pin, void* Pout, void* transform,
                    unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_vector_masked<Wide<const Vec3>, UniformAsWide<const Matrix44>>(
        Pin, Pout, transform, mask_transform, mask_value);
}


OSL_BATCHOP void
__OSL_MASKED_OP3(transform_vector, Wdv, Wdv,
                 m)(void* Pin, void* Pout, void* transform,
                    unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_vector_masked<Wide<const Dual2<Vec3>>,
                                 UniformAsWide<const Matrix44>>(Pin, Pout,
                                                                transform,
                                                                mask_transform,
                                                                mask_value);
}

namespace {
template<typename InputAccessorT>
OSL_FORCEINLINE void
impl_transform_normal_masked(void* Pin, void* Pout, Wide<const Matrix44> wM,
                             unsigned int mask_transform,
                             unsigned int mask_value)
{
    typedef typename InputAccessorT::NonConstValueType DataType;

    Mask mask(mask_value);
    Mask succeeded(mask_transform);

    // only operate on active lanes
    Mask activeMask = mask & succeeded;

    Masked<DataType> wresult(Pout, activeMask);
    InputAccessorT inPoints(Pin);

    // Transform with Normal semantics

    Block<int> notAffineBlock;
    Wide<int> wnotAffine(notAffineBlock);

    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            DataType v     = inPoints[lane];
            Matrix44 M     = wM[lane];
            bool is_affine = true;
            if (wresult.mask()[lane]) {
                is_affine = test_if_affine(M);
                if (is_affine) {
                    wresult[ActiveLane(lane)] = multiplyDirByMatrix(
                        inlinedTransposed(OSL::affineInverse(M)), v);
                }
            }
            wnotAffine[lane] = (!is_affine);  // false when lane is masked off
        }
    }

    if (testIfAnyLaneIsNonZero(wnotAffine)) {
        invoke([=]() -> void {
            // DO NOT VECTORIZE the slow path
            for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
                if (wnotAffine[lane]) {
                    OSL_DASSERT(wresult.mask().is_on(lane));

                    DataType v       = inPoints[lane];
                    const Matrix44 M = wM[lane];

                    //M.inverse().transposed().multDirMatrix (v, r);
                    // Use helper that has specializations for Dual2
                    wresult[ActiveLane(lane)] = multiplyDirByMatrix(
                        inlinedTransposed(nonAffineInverse(M)), v);
                }
            }
        });
    }

    impl_copy_untransformed_lanes(inPoints, Pout, succeeded, mask);
}

template<typename InputAccessorT>
OSL_FORCEINLINE void
impl_transform_normal_masked(void* Pin, void* Pout, const Matrix44& M,
                             unsigned int mask_transform,
                             unsigned int mask_value)
{
    typedef typename InputAccessorT::NonConstValueType DataType;

    Mask mask(mask_value);
    Mask succeeded(mask_transform);

    // only operate on active lanes
    Mask activeMask = mask & succeeded;

    Masked<DataType> wresult(Pout, activeMask);
    InputAccessorT inPoints(Pin);

    // Transform with Normal semantics

    Matrix44 invM { Imath::UNINITIALIZED };
    if (OSL_UNLIKELY(!test_if_affine(M))) {
        // Isolate expensive unlikely code
        // in its own function as to not influence
        // the optimization of the fast path
        invoke([&]() -> void { invM = OSL::nonAffineInverse(M); });
    } else {
        invM = OSL::affineInverse(M);
    }
    OSL_FORCEINLINE_BLOCK
    {
        Matrix44 normalTransform = inlinedTransposed(invM);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            // Should InputAccessorT be UniformAsWide<DataType>,
            // we expect the compiler to hoist the next 2 lines
            // outside the loops and keep the broadcasting
            // of the result as SIMD
            DataType v      = inPoints[lane];
            DataType result = multiplyDirByMatrix(normalTransform, v);

            wresult[lane] = result;
        }
    }

    impl_copy_untransformed_lanes(inPoints, Pout, succeeded, mask);
}

}  // namespace

OSL_BATCHOP void
__OSL_MASKED_OP3(transform_normal, v, Wv,
                 m)(void* Pin, void* Pout, void* transform,
                    unsigned int mask_transform, unsigned int mask_value)
{
    // TODO: see if we can get gen_transform to call the vvm version then do a masked broadcast
    impl_transform_normal_masked<UniformAsWide<const Vec3>>(Pin, Pout,
                                                            MAT(transform),
                                                            mask_transform,
                                                            mask_value);
}


OSL_BATCHOP void
__OSL_MASKED_OP3(transform_normal, v, Wv,
                 Wm)(void* Pin, void* Pout, void* transform,
                     unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_normal_masked<UniformAsWide<const Vec3>>(
        Pin, Pout, Wide<const Matrix44>(transform), mask_transform, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_normal, Wv, Wv,
                 Wm)(void* Pin, void* Pout, void* transform,
                     unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_normal_masked<Wide<const Vec3>>(
        Pin, Pout, Wide<const Matrix44>(transform), mask_transform, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_normal, Wdv, Wdv,
                 Wm)(void* Pin, void* Pout, void* transform,
                     unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_normal_masked<Wide<const Dual2<Vec3>>>(
        Pin, Pout, Wide<const Matrix44>(transform), mask_transform, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_normal, Wv, Wv,
                 m)(void* Pin, void* Pout, void* transform,
                    unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_normal_masked<Wide<const Vec3>>(Pin, Pout, MAT(transform),
                                                   mask_transform, mask_value);
}



OSL_BATCHOP void
__OSL_MASKED_OP3(transform_normal, Wdv, Wdv,
                 m)(void* Pin, void* Pout, void* transform,
                    unsigned int mask_transform, unsigned int mask_value)
{
    impl_transform_normal_masked<Wide<const Dual2<Vec3>>>(Pin, Pout,
                                                          MAT(transform),
                                                          mask_transform,
                                                          mask_value);
}



OSL_BATCHOP void
__OSL_OP2(determinant, Wf, Wm)(void* wr_, void* wm_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wm(wm_);
        Wide<float> wr(wr_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 m = wm[lane];
            float r    = det4x4(m);
            wr[lane]   = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(determinant, Wf, Wm)(void* wr_, void* wm_,
                                      unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Matrix44> wm(wm_);
        Masked<float> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Matrix44 m = wm[lane];
            float r    = det4x4(m);
            wr[lane]   = r;
        }
    }
}

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
