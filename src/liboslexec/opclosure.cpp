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



static void
parse_keyword_args(const ClosureRegistry::ClosureEntry *clentry, VaryingRef<ClosureComponent *> &comp,
                   ShadingExecution *exec, const int *args, int nattrs)
{

    for (int attr_i = 0; attr_i < nattrs; ++attr_i) {
        int argno = attr_i * 2;
        Symbol &Key (exec->sym (args[argno]));
        Symbol &Value (exec->sym (args[argno + 1]));

        ASSERT(Key.typespec().is_string());
        ASSERT(Key.is_constant());
        ustring key;
        SHADE_LOOP_BEGIN
            key = * (ustring *) Key.data(i);
            break;
        SHADE_LOOP_END

        TypeDesc td = Value.typespec().simpletype();
        bool legal = false;
        // Make sure there is some keyword arg that has the name and the type
        for (int t = 0; t < clentry->nkeyword; ++t) {
            const ClosureParam &param = clentry->params[clentry->nformal + t];
            if (param.type == td && !strcmp(key.c_str(), param.key))
                legal = true;
        }
        if (legal) {
            SHADE_LOOP_BEGIN
                ClosureComponent::Attr *attr = comp[i]->attrs() + attr_i;
                attr->key = key;
                char *value = (char *)Value.data() + i*Value.step();
                memcpy(&attr->value, value, td.size());
            SHADE_LOOP_END
        }
    }
}



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

    ASSERT(nargs >= (2 + clentry->nformal));
    int nattrs = (nargs - 2 - clentry->nformal) / 2;

    VaryingRef<ClosureComponent *> result ((ClosureComponent **)Result.data(), Result.step());
    SHADE_LOOP_BEGIN
        ClosureComponent *comp = exec->context()->closure_component_allot(clentry->id, clentry->struct_size, nattrs);
        if (clentry->prepare)
            clentry->prepare(exec->renderer(), clentry->id, comp->mem);
        else
            memset(comp->mem, 0, clentry->struct_size);
        for (size_t carg = 0; carg < clentry->params.size(); ++carg) {
            const ClosureParam &p = clentry->params[carg];
            if (p.key != NULL) break;
            ASSERT(p.offset < clentry->struct_size);
            ASSERT(carg < (size_t)clentry->nformal);
            write_closure_param(p.type, comp->mem, p.offset, carg + 2, i, exec, args);
        }
        if (clentry->setup)
            clentry->setup(exec->renderer(), clentry->id, comp->mem);
        result[i] = comp;
    SHADE_LOOP_END

    if (nattrs && clentry->nkeyword)
        parse_keyword_args(clentry, result, exec, args + clentry->nformal + 2, nattrs);
}




}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif



#if 1

extern "C" const ClosureColor *
osl_add_closure_closure (SingleShaderGlobal *sg,
                         const ClosureColor *a, const ClosureColor *b)
{
    if (a == NULL)
        return b;
    else if (b == NULL)
        return a;
    return sg->context->closure_add_allot (a, b);
}


extern "C" const ClosureColor *
osl_mul_closure_color (SingleShaderGlobal *sg, ClosureColor *a, const Color3 *w)
{
    if (a == NULL) return NULL;
    return sg->context->closure_mul_allot (*w, a);
}


extern "C" const ClosureColor *
osl_mul_closure_float (SingleShaderGlobal *sg, ClosureColor *a, float w)
{
    if (a == NULL) return NULL;
    return sg->context->closure_mul_allot (w, a);
}


extern "C" ClosureComponent *
osl_allocate_closure_component (SingleShaderGlobal *sg, int id, int size, int nattrs)
{
    return sg->context->closure_component_allot(id, size, nattrs);
}



extern "C" const char *
osl_closure_to_string (SingleShaderGlobal *sg, ClosureColor *c)
{
    // Special case for printing closures
    std::stringstream stream;
    print_closure(stream, c, &sg->context->shadingsys());
    return ustring(stream.str ()).c_str();
}

#endif
