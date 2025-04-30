// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

template<typename F>
BSDL_INLINE F
sample_cdf(const F* data, unsigned int n, F x, unsigned int* idx, F* pdf)
{
    assert(x >= 0);
    assert(x < 1);
    *idx = static_cast<unsigned int>(asl::upper_bound(data, data + n, x)
                                     - data);
    assert(*idx < n);
    assert(x < data[*idx]);
    F scaled_sample;
    if (*idx == 0) {
        *pdf          = data[0];
        scaled_sample = Sample::stretch(x, static_cast<F>(0), data[0]);
    } else {
        assert(x >= data[*idx - 1]);
        *pdf          = data[*idx] - data[*idx - 1];
        scaled_sample = Sample::stretch(x, data[*idx - 1], *pdf);
    }
    return scaled_sample;
}

// For small fixed size CDFs that we hope can fit in registers and where we
// avoid any random access.
template<int N> struct StaticCdf {
    // To avoid a dynamic search with a for loop that can't be unrolled, we
    // implement the binary search with a recursive template.
    struct result {
        int idx;
        float lo, hi;
    };
    // Recursive rule
    template<int S, int M> struct searcher {
        // Search by a random number r
        BSDL_INLINE_METHOD result operator()(const float* cdf, float r) const
        {
            constexpr int H = M >> 1;
            return r < cdf[S + H - 1] ? searcher<S, H>()(cdf, r)
                                      : searcher<S + H, M - H>()(cdf, r);
        }
        // Search by index, equivalent to [i] but trading random access for
        // small branches
        BSDL_INLINE_METHOD result operator()(const float* cdf, int i) const
        {
            constexpr int H = M >> 1;
            return i < S + H ? searcher<S, H>()(cdf, i)
                             : searcher<S + H, M - H>()(cdf, i);
        }
    };
    // Basic cases
    template<int S> struct searcher<S, 1> {
        BSDL_INLINE_METHOD result operator()(const float* cdf, float r) const
        {
            return { S, S ? cdf[S - 1] : 0.0f, cdf[S] };
        }
        BSDL_INLINE_METHOD result operator()(const float* cdf, int i) const
        {
            return { S, S ? cdf[S - 1] : 0.0f, cdf[S] };
        }
    };

    BSDL_INLINE_METHOD float build()
    {
        // These loops are easily unrolled
        for (int i = 1; i < N; ++i)
            cdf[i] += cdf[i - 1];
        const float total = cdf[N - 1];
        if (total > 0) {
            for (int i = 0; i < N; ++i)
                cdf[i] /= total;
            // Watch out for fast math crushing denormals and not
            // dividing correctly.
            const float top = cdf[N - 1];
            for (int i = N - 1; i >= 0 && cdf[i] == top; --i)
                cdf[i] = 1;
        }
        return total;
    }
    BSDL_INLINE_METHOD float sample(float x, int* idx, float* pdf)
    {
        // We have to do it this way, get idx, lo and hi from the search because
        // if we get only idx and fetch lo and hi from it, that's random access
        // and CUDA sends the whole CDF to local memory.
        auto res = searcher<0, N>()(cdf, x);
        *idx     = res.idx;
        *pdf     = res.hi - res.lo;
        return Sample::stretch(x, res.lo, *pdf);
    }
    BSDL_INLINE_METHOD float pdf(int i) const
    {
        auto res = searcher<0, N>()(cdf, i);
        return res.hi - res.lo;
    }
    // These can potentially produce random access if not called with constants
    // or unrollable loop variables.
    BSDL_INLINE_METHOD float& operator[](int i) { return cdf[i]; }
    BSDL_INLINE_METHOD const float& operator[](int i) const { return cdf[i]; }

    float cdf[N];
};

BSDL_LEAVE_NAMESPACE
