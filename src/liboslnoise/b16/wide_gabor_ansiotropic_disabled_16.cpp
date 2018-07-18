/*
Copyright (c) 2012 Sony Pictures Imageworks Inc., et al.
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

#include "../wide_gabornoise.h"


OSL_NAMESPACE_ENTER

namespace pvt {


template void
wide_gabor<1 /*ansiotropic*/, DisabledFilterPolicy, 16 /* WidthT */>(
        MaskedAccessor<Dual2<float>,16 /*WidthT*/> wResult,
        ConstWideAccessor<Dual2<float>, 16 /*WidthT*/> wX,
        NoiseParams const *opt);

template void
wide_gabor<1 /*ansiotropic*/, DisabledFilterPolicy, 16 /* WidthT */>(
        MaskedAccessor<Dual2<float>,16 /*WidthT*/> wResult,
        ConstWideAccessor<Dual2<float>, 16 /*WidthT*/> wX,
        ConstWideAccessor<Dual2<float>, 16 /*WidthT*/> wY,
        NoiseParams const *opt);

template void
wide_gabor<1 /*ansiotropic*/, DisabledFilterPolicy, 16 /* WidthT */>(
        MaskedAccessor<Dual2<float>,16 /*WidthT*/> wResult,
        ConstWideAccessor<Dual2<Vec3>, 16 /*WidthT*/> wP,
        NoiseParams const *opt);


template void
wide_pgabor<1 /*ansiotropic*/, DisabledFilterPolicy, 16 /* WidthT */>(
        MaskedAccessor<Dual2<float>,16 /*WidthT*/> wResult,
        ConstWideAccessor<Dual2<float>, 16 /*WidthT*/> wX,
        ConstWideAccessor<float, 16 /*WidthT*/> wXp,
        NoiseParams const *opt);

template void
wide_pgabor<1 /*ansiotropic*/, DisabledFilterPolicy, 16 /* WidthT */>(
        MaskedAccessor<Dual2<float>,16 /*WidthT*/> wResult,
        ConstWideAccessor<Dual2<float>, 16 /*WidthT*/> wX,
        ConstWideAccessor<Dual2<float>, 16 /*WidthT*/> wY,
        ConstWideAccessor<float, 16 /*WidthT*/> wXp,
        ConstWideAccessor<float, 16 /*WidthT*/> wYp,
        NoiseParams const *opt);

template void
wide_pgabor<1 /*ansiotropic*/, DisabledFilterPolicy, 16 /* WidthT */>(
        MaskedAccessor<Dual2<float>,16 /*WidthT*/> wResult,
        ConstWideAccessor<Dual2<Vec3>, 16 /*WidthT*/> wP,
        ConstWideAccessor<Vec3, 16 /*WidthT*/> wPp,
        NoiseParams const *opt);




}; // namespace pvt
OSL_NAMESPACE_EXIT
