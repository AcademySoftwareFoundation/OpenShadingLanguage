// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <iostream>
#include <cmath>

#include <OSL/genclosure.h>
#include <OSL/batched_shaderglobals.h>

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
    Wide<ClosureColorPtr> wide_out (wide_out_);
    Wide<ClosureColorPtr> wide_closure_a (wide_closure_a_);
    Wide<ClosureColorPtr> wide_closure_b (wide_closure_b_);
    Mask mask (mask_value);
    
    ClosureAdd *add = bsg->uniform.context->batched<__OSL_WIDTH>().closure_add_allot ();

    // Currently this is done as AOS, future work may improve this by converting to SOA
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        if (mask[lane]) {
            ClosureAdd *a = &add[lane];
            a->id = ClosureColor::ADD;
            a->closureA = wide_closure_a[lane];
            a->closureB = wide_closure_b[lane];
        }
    }
    
    // The input closures may be aliased with the output, so we
    // do the write-back to the output after we've set-up the
    // new result
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        if (mask[lane]) {
            ClosureColorPtr a = (ClosureColorPtr)&add[lane];
            wide_out[lane] = a;
        }
    }
}

OSL_BATCHOP void
__OSL_MASKED_OP(mul_closure_color) (void *bsg_, void *wide_out_,  
        void *wide_closure_a_,  void *wide_weight_, unsigned int mask_value)
{
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);
    // TODO avoid allot if weight = 0 or closure = null?
    Wide<ClosureColorPtr> wide_out (wide_out_);
    Wide<Color3>          wide_weight (wide_weight_);
    Wide<ClosureColorPtr> wide_closure_a (wide_closure_a_);
    Mask mask(mask_value);

    ClosureMul *mul = bsg->uniform.context->batched<__OSL_WIDTH>().closure_mul_allot();

    // Currently this is done as AOS, future work may improve this by converting to SOA
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
         if (mask[lane]) {
           ClosureMul *m = &mul[lane];
           m->id = ClosureColor::MUL;
           m->weight = wide_weight[lane];
           m->closure = wide_closure_a[lane];
        }
    }
    
    // The input closure may be aliased with the output, so we
    // do the write-back to the output after we've set-up the
    // new result
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        if (mask[lane]) {
            ClosureColorPtr m = (ClosureColorPtr)&mul[lane];
            wide_out[lane] = m;
        }
    }
}


OSL_BATCHOP void
__OSL_MASKED_OP(mul_closure_float) (void *bsg_, void *wide_out_,  
        void *wide_closure_a_, void * wide_weight_, unsigned int mask_value)
{
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);
    // TODO avoid allot if weight = 0 or closure = null?
    Wide<ClosureColorPtr> wide_out (wide_out_);
    Wide<float>           wide_weight (wide_weight_);
    Wide<ClosureColorPtr> wide_closure_a (wide_closure_a_);
    Mask mask(mask_value);

    ClosureMul *mul = bsg->uniform.context->batched<__OSL_WIDTH>().closure_mul_allot ();

    // Currently this is done as AOS, future work may improve this by converting to SOA
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        if (mask[lane]) {
           ClosureMul *m = &mul[lane];
           m->id = ClosureColor::MUL;
           float w = wide_weight[lane];
           m->weight.setValue (w,w,w);
           m->closure = wide_closure_a[lane];
        }
    }
    
    // The input closure may be aliased with the output, so we
    // do the write-back to the output after we've set-up the
    // new result
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        if (mask[lane]) {
            ClosureColorPtr m = (ClosureColorPtr)&mul[lane];
            wide_out[lane] = m;
        }
    }
}


void init_closure_component(Masked<ClosureComponentPtr> &wComp, int id, int size, Wide<Color3> wWeight, ClosureComponent* comp_mem)
{

    constexpr int alignment = alignof(ClosureComponent);
    size_t stride = (int)((sizeof(ClosureComponent) + size + alignment-1)/alignment)*alignment;
    // Currently this is done as AOS, future work may improve this by converting to SOA
    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for(int lane = 0; lane < __OSL_WIDTH; ++lane) {
        if (wComp.mask()[lane]) {
            ClosureComponent * comp = (ClosureComponent *)((char*)comp_mem + lane * stride);
            wComp[lane] = comp;
            comp->id = id;
            comp->w = wWeight[lane];
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
    Wide<Color3> wWeight (&one_block);
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
    Wide<Color3> wWeight (wide_weight_);
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
