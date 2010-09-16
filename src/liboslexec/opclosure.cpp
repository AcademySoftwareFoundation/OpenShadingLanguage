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

#include <cmath>

#include "oslops.h"
#include "oslexec_pvt.h"
#include "genclosure.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {



// implemented in builtin_closures.cpp
void setup_builtin_closure(int id, void *data);



namespace pvt {



DECLOP (OP_closure)
{
    ASSERT (nargs >= 2); // at least the result and the ID

    Symbol &Result (exec->sym (args[0]));
    DASSERT(Result.typespec().is_closure() && Result.is_varying());
    Symbol &Id (exec->sym (args[1]));
    DASSERT(Id.typespec().is_string() && Id.is_uniform());
    VaryingRef<ustring> closure_name(Id.data(), Id.step());
    const ClosureRegistry::ClosureEntry * clentry = NULL;
    clentry = exec->shadingsys()->find_closure(closure_name[exec->beginpoint()]);
    ASSERT(clentry);

    VaryingRef<ustring> labels[ClosurePrimitive::MAXCUSTOM+1];
    int nlabels = 0;
    int nformal_params = nargs;
    for (int tok = 2; tok < (nargs - 1); tok ++) {
        Symbol &Name (exec->sym (args[tok]));
        if (Name.typespec().is_string()) {
             nformal_params = std::min(nformal_params, tok);
             ustring name = * (ustring *) Name.data();
             Symbol &Val (exec->sym (args[tok + 1]));
             if (Val.typespec().is_string()) {
                 if (name == Strings::label) {
                     if (nlabels == ClosurePrimitive::MAXCUSTOM)
                         exec->error ("Too many labels to closure (%s:%d)",
                                       exec->op().sourcefile().c_str(),
                                       exec->op().sourceline());
                     else {
                         labels[nlabels].init((ustring*) Val.data(), Val.step());
                         nlabels++;
                         tok++;
                     }
                 } else {
                     exec->error ("Unknown closure optional argument: \"%s\", <%s> (%s:%d)",
                                     name.c_str(),
                                     Val.typespec().c_str(),
                                     exec->op().sourcefile().c_str(),
                                     exec->op().sourceline());
                 }
            } else {
                 exec->error ("Malformed keyword args to closure (%s:%d)",
                              exec->op().sourcefile().c_str(),
                              exec->op().sourceline());
            }
        }
    }
    // From now on, the keyword arguments don't exist
    nargs = nformal_params;

    VaryingRef<ClosureColor *> result ((ClosureColor **)Result.data(), Result.step());
    SHADE_LOOP_BEGIN
        char* mem = result[i]->allocate_component (clentry->id, clentry->struct_size);
        if (clentry->prepare)
            clentry->prepare(exec->renderer(), clentry->id, mem);
        else
            memset(mem, 0, clentry->struct_size);
        for (size_t carg = 0; carg < clentry->params.size(); ++carg)
        {
            const ClosureParam &p = clentry->params[carg];
            ASSERT(p.offset < clentry->struct_size);
            write_closure_param(p.type, mem, p.offset, carg + 2, i, exec, nargs, args);
        }
        if (clentry->labels_offset >= 0)
        {
            int l;
            for (l = 0; l < nlabels && l < clentry->max_labels; ++l)
                ((ustring *)(mem + clentry->labels_offset))[l] = labels[l][i];
            ((ustring *)(mem + clentry->labels_offset))[l] = Labels::NONE;
        }
        if (clentry->setup)
            clentry->setup(exec->renderer(), clentry->id, mem);
    SHADE_LOOP_END
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif



#if 1
// Some wrapper functions we need to call from the LLVM-generated code.

extern "C" void *
osl_closure_allot (SingleShaderGlobal *sg)
{
    ShadingContext *ctx = (ShadingContext *)sg->context;
    ClosureColor *r = ctx->closure_ptr_allot ();
    DASSERT (r && "bad closure allot");
    r->clear ();
    return r;
}


extern "C" void
osl_closure_clear (ClosureColor *r)
{
    r->clear ();
}


extern "C" void
osl_closure_clear_indexed (ClosureColor **r, int i)
{
    r[i]->clear ();
}


extern "C" void
osl_closure_assign (ClosureColor *r, ClosureColor *x)
{
    DASSERT (r);  DASSERT (x);
    if (r != x)
      *r = *x;
}


extern "C" void
osl_closure_assign_indexed (ClosureColor **r, int ri,
                            const ClosureColor **x, int xi)
{
    *(r[ri]) = *(x[xi]);
}


extern "C" void
osl_add_closure_closure (SingleShaderGlobal *sg, ClosureColor *r,
                         const ClosureColor *a, const ClosureColor *b)
{
    r->add (*a, *b, &sg->context->shadingsys());
}


extern "C" void
osl_mul_closure_color (ClosureColor *r, ClosureColor *a, const Color3 *b)
{
    if (r != a) *r = *a;
    *r *= *b;
}


extern "C" void
osl_mul_closure_float (ClosureColor *r, ClosureColor *a, float b)
{
    if (r != a) *r = *a;
    *r *= b;
}


extern "C" void *
osl_allocate_closure_component (ClosureColor *r, int id, int size)
{
    DASSERT (r);
    void *mem = r->allocate_component (id, size);
    return mem;
}


extern "C" const char *
osl_closure_to_string (ClosureColor *c)
{
    // Special case for printing closures
    std::stringstream stream;
    stream << *c;
    return ustring(stream.str ()).c_str();
}

#endif
