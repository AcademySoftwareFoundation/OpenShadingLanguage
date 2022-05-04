// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <iostream>
#include <cmath>

#include <OSL/genclosure.h>
#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

#include "oslexec_pvt.h"

#include "define_opname_macros.h"

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)


OSL_BATCHOP void
__OSL_MASKED_OP(add_closure_closure) (void *bsg_, void *wide_out_, 
        void *wide_closure_a_, void* wide_closure_b_, unsigned int mask_value)
{
    // TODO avoid allot if !a or !b?
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);
    Mask mask(mask_value);
    Masked<ClosureColorPtr> wide_out (wide_out_, mask);
    Wide<const ClosureColorPtr> wide_closure_a (wide_closure_a_);
    Wide<const ClosureColorPtr> wide_closure_b (wide_closure_b_);
    
    ClosureAdd *add = bsg->uniform.context->batched<__OSL_WIDTH>().closure_add_allot ();

    // Currently this is done as AOS, future work may improve this by converting to SOA
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        ClosureAdd *a = &add[lane];
        const ClosureColor* ca = wide_closure_a[lane];
        const ClosureColor* cb = wide_closure_b[lane];
        if (mask[lane]) {
            a->id = ClosureColor::ADD;
            a->closureA = ca;
            a->closureB = cb;
        }
    }
    
    // The input closures may be aliased with the output, so we
    // do the write-back to the output after we've set-up the
    // new result
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        ClosureColorPtr a = (ClosureColorPtr)&add[lane];
        wide_out[lane] = a;
    }
}

OSL_BATCHOP void
__OSL_MASKED_OP(mul_closure_color) (void *bsg_, void *wide_out_,  
        void *wide_closure_a_,  void *wide_weight_, unsigned int mask_value)
{
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);
    // TODO avoid allot if weight = 0 or closure = null?
    Mask mask(mask_value);
    Masked<ClosureColorPtr> wide_out (wide_out_, mask);
    Wide<const Color3> wide_weight (wide_weight_);
    Wide<const ClosureColorPtr> wide_closure_a (wide_closure_a_);
    
    ClosureMul *mul = bsg->uniform.context->batched<__OSL_WIDTH>().closure_mul_allot();

    // Currently this is done as AOS, future work may improve this by converting to SOA
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        ClosureMul *m = &mul[lane];
        const Color3 w = wide_weight[lane];
        const ClosureColor* ca = wide_closure_a[lane]; 
        if (mask[lane]) {
            m->id = ClosureColor::MUL;
            m->weight = w;
            m->closure = ca;
        }
    }
    
    // The input closure may be aliased with the output, so we
    // do the write-back to the output after we've set-up the
    // new result
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        ClosureColorPtr m = (ClosureColorPtr)&mul[lane];
        wide_out[lane] = m;
    }
}


OSL_BATCHOP void
__OSL_MASKED_OP(mul_closure_float) (void *bsg_, void *wide_out_,  
        void *wide_closure_a_, void * wide_weight_, unsigned int mask_value)
{
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);
    // TODO avoid allot if weight = 0 or closure = null?
    Mask mask(mask_value);
    Masked<ClosureColorPtr> wide_out (wide_out_, mask);
    Wide<const float> wide_weight (wide_weight_);
    Wide<const ClosureColorPtr> wide_closure_a (wide_closure_a_);

    ClosureMul *mul = bsg->uniform.context->batched<__OSL_WIDTH>().closure_mul_allot ();

    // Currently this is done as AOS, future work may improve this by converting to SOA
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        ClosureMul *m = &mul[lane];
        const float w = wide_weight[lane];
        const Color3 weight(w, w, w);
        const ClosureColor* ca = wide_closure_a[lane]; 
        if (mask[lane]) {
            m->id = ClosureColor::MUL;
            m->weight = weight;
            m->closure = ca;
        }
    }
    
    // The input closure may be aliased with the output, so we
    // do the write-back to the output after we've set-up the
    // new result
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        ClosureColorPtr m = (ClosureColorPtr)&mul[lane];
        wide_out[lane] = m;
    }
}


void init_closure_component(Masked<ClosureComponentPtr> &wComp, int id, int size, Wide<const Color3> wWeight, ClosureComponent* comp_mem)
{

    constexpr int alignment = alignof(ClosureComponent);
    size_t stride = (int)((sizeof(ClosureComponent) + size + alignment-1)/alignment)*alignment;
    // Currently this is done as AOS, future work may improve this by converting to SOA
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for(int lane = 0; lane < __OSL_WIDTH; ++lane) {
        ClosureComponent * comp = (ClosureComponent *)((char*)comp_mem + lane * stride);
        const Color3 w = wWeight[lane];
        if (wComp.mask()[lane]) {
            wComp[lane] = comp;
            comp->id = id;
            comp->w = w;
        }
    }
}

OSL_BATCHOP void
__OSL_MASKED_OP(allocate_closure_component) (void *bsg_, void *wide_out_, int id, int size, unsigned int mask_value)
{
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);
    
    Masked<ClosureComponentPtr> wComp(wide_out_, Mask(mask_value));
    Block<Color3> one_block;
    assign_all(one_block, Color3(1.0f));
    Wide<const Color3> wWeight (&one_block);
    ClosureComponent *comp_mem = bsg->uniform.context->batched<__OSL_WIDTH>().closure_component_allot (size);
    init_closure_component (wComp, id, size, wWeight, comp_mem);
}

OSL_BATCHOP void
__OSL_MASKED_OP(allocate_weighted_closure_component) (void *bsg_, void *wide_out_, 
        int id, int size, void * wide_weight_, unsigned int mask_value)
{
    // TODO return nullptr if all are 0 or masked?
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);
    Masked<ClosureComponentPtr> wComp (wide_out_, Mask(mask_value));
    Wide<const Color3> wWeight (wide_weight_);
    ClosureComponent *comp_mem = bsg->uniform.context->batched<__OSL_WIDTH>().closure_component_allot (size);
    init_closure_component (wComp, id, size, wWeight, comp_mem);
}

// This currently duplicates the scalar version of the op, but
// accesses the context through BatchedShaderGlobals instead of ShaderGlobals
// future work would extend this to operate on a whole batch of closures at once
OSL_BATCHOP const char *
__OSL_OP(closure_to_string) (void *bsg_, ClosureColor *c)
{
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);
    // Special case for printing closures
    std::ostringstream stream;
    stream.imbue (std::locale::classic());  // force C locale
    print_closure (stream, c, &bsg->uniform.context->shadingsys());
    return ustring (stream.str ()).c_str();
}


} // namespace pvt
OSL_NAMESPACE_EXIT
