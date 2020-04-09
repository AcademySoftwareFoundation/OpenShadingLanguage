// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <OSL/dual_vec.h>
#include <OSL/oslconfig.h>
#include <algorithm> // upper_bound

OSL_NAMESPACE_ENTER

struct Background {
    Background() : values(0), rows(0), cols(0) {}
    ~Background() {
        delete [] values;
        delete [] rows;
        delete [] cols;
    }

    template <typename F, typename T>
    void prepare(int resolution, F cb, T* data) {
        res = resolution;
        if (res < 32) res = 32; // validate
        invres = 1.0f / res;
        invjacobian = res * res / float(4 * M_PI);
        values = new Vec3[res * res];
        rows   = new float[res];
        cols   = new float[res * res];
        for (int y = 0, i = 0; y < res; y++) {
            for (int x = 0; x < res; x++, i++) {
                values[i] = cb(map(x + 0.5f, y + 0.5f), data);
                cols[i] = std::max(std::max(values[i].x, values[i].y), values[i].z) + ((x > 0) ? cols[i - 1] : 0.0f);
            }
            rows[y] = cols[i - 1] + ((y > 0) ? rows[y - 1] : 0.0f);
            // normalize the pdf for this scanline (if it was non-zero)
            if (cols[i - 1] > 0)
                for (int x = 0; x < res; x++)
                    cols[i - res + x] /= cols[i - 1];
        }
        // normalize the pdf across all scanlines
        for (int y = 0; y < res; y++)
            rows[y] /= rows[res - 1];

        // both eval and sample below return a "weight" that is
        // value[i] / row*col_pdf, so might as well bake it into the table
        for (int y = 0, i = 0; y < res; y++) {
            float row_pdf = rows[y] - (y > 0 ? rows[y - 1] : 0.0f);
            for (int x = 0; x < res; x++, i++) {
                float col_pdf = cols[i] - (x > 0 ? cols[i - 1] : 0.0f);
                values[i] /= row_pdf * col_pdf * invjacobian;
            }
        }
#if 0  // DEBUG: visualize importance table
        using namespace OIIO;
        ImageOutput* out = ImageOutput::create("bg.exr");
        ImageSpec spec(res, res, 3, TypeDesc::TypeFloat);
        if (out && out->open("bg.exr", spec))
            out->write_image(TypeDesc::TypeFloat, &values[0]);
        delete out;
#endif
    }

    Vec3 eval(const Vec3& dir, float& pdf) const {
        // map from sphere to unit-square
        float u = OIIO::fast_atan2(dir.y, dir.x) * float(M_1_PI * 0.5f);
        if (u < 0) u++;
        float v = (1 - dir.z) * 0.5f;
        // retrieve nearest neighbor
        int x = (int) (u * res); if (x < 0) x = 0; else if (x >= res) x = res - 1;
        int y = (int) (v * res); if (y < 0) y = 0; else if (y >= res) y = res - 1;
        int i = y * res + x;
        float row_pdf = rows[y] - (y > 0 ? rows[y - 1] : 0.0f);
        float col_pdf = cols[i] - (x > 0 ? cols[i - 1] : 0.0f);
        pdf = row_pdf * col_pdf * invjacobian;
        return values[i];
    }

    Vec3 sample(float rx, float ry, Dual2<Vec3>& dir, float& pdf) const {
        float row_pdf, col_pdf;
        unsigned x, y;
        ry = sample_cdf(rows, res, ry, &y, &row_pdf);
        rx = sample_cdf(cols + y * res, res, rx, &x, &col_pdf);
        dir = map(x + rx, y + ry);
        pdf = row_pdf * col_pdf * invjacobian;
        return values[y * res + x];
    }

private:
    Dual2<Vec3> map(float x, float y) const {
        // pixel coordinates of entry (x,y)
        Dual2<float> u = Dual2<float>(x, 1, 0) * invres;
        Dual2<float> v = Dual2<float>(y, 0, 1) * invres;
        Dual2<float> theta   = u * float(2 * M_PI);
        Dual2<float> st, ct;
        fast_sincos(theta, &st, &ct);
        Dual2<float> cos_phi = 1.0f - 2.0f * v;
        Dual2<float> sin_phi = sqrt(1.0f - cos_phi * cos_phi);
        return make_Vec3(sin_phi * ct,
                         sin_phi * st,
                         cos_phi);
    }

    static float sample_cdf(const float* data, unsigned int n, float x, unsigned int *idx, float* pdf) {
        OSL_DASSERT(x >= 0);
        OSL_DASSERT(x < 1);
        *idx = std::upper_bound(data, data + n, x) - data;
        OSL_DASSERT(*idx < n);
        OSL_DASSERT(x < data[*idx]);
        float scaled_sample;
        if (*idx == 0) {
            *pdf = data[0];
            scaled_sample = x / data[0];
        } else {
            OSL_DASSERT(x >= data[*idx - 1]);
            *pdf = data[*idx] - data[*idx - 1];
            scaled_sample = (x - data[*idx - 1]) / (data[*idx] - data[*idx - 1]);
        }
        // keep result in [0,1)
        return std::min(scaled_sample, 0.99999994f);
    }

    Vec3*  values;  // actual map
    float* rows;    // probability of choosing a given row 'y'
    float* cols;    // probability of choosing a given column 'x', given that we've chosen row 'y'
    int res;        // resolution in pixels of the precomputed table
    float invres;   // 1 / resolution
    float invjacobian;
};

OSL_NAMESPACE_EXIT
