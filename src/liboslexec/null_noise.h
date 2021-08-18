// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

#include <OSL/dual_vec.h>

OSL_NAMESPACE_ENTER
namespace pvt {


struct NullNoise {
    OSL_HOSTDEVICE NullNoise() {}
    OSL_HOSTDEVICE inline void operator()(float& result, float x) const
    {
        result = 0.0f;
    }
    OSL_HOSTDEVICE inline void operator()(float& result, float x, float y) const
    {
        result = 0.0f;
    }
    OSL_HOSTDEVICE inline void operator()(float& result, const Vec3& p) const
    {
        result = 0.0f;
    }
    OSL_HOSTDEVICE inline void operator()(float& result, const Vec3& p,
                                          float t) const
    {
        result = 0.0f;
    }
    OSL_HOSTDEVICE inline void operator()(Vec3& result, float x) const
    {
        result = v();
    }
    OSL_HOSTDEVICE inline void operator()(Vec3& result, float x, float y) const
    {
        result = v();
    }
    OSL_HOSTDEVICE inline void operator()(Vec3& result, const Vec3& p) const
    {
        result = v();
    }
    OSL_HOSTDEVICE inline void operator()(Vec3& result, const Vec3& p,
                                          float t) const
    {
        result = v();
    }
    OSL_HOSTDEVICE inline void
    operator()(Dual2<float>& result, const Dual2<float>& x, int seed = 0) const
    {
        result.set(0.0f, 0.0f, 0.0f);
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<float>& result,
                                          const Dual2<float>& x,
                                          const Dual2<float>& y,
                                          int seed = 0) const
    {
        result.set(0.0f, 0.0f, 0.0f);
    }
    OSL_HOSTDEVICE inline void
    operator()(Dual2<float>& result, const Dual2<Vec3>& p, int seed = 0) const
    {
        result.set(0.0f, 0.0f, 0.0f);
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<float>& result,
                                          const Dual2<Vec3>& p,
                                          const Dual2<float>& t,
                                          int seed = 0) const
    {
        result.set(0.0f, 0.0f, 0.0f);
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<Vec3>& result,
                                          const Dual2<float>& x) const
    {
        result.set(v(), v(), v());
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<Vec3>& result,
                                          const Dual2<float>& x,
                                          const Dual2<float>& y) const
    {
        result.set(v(), v(), v());
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<Vec3>& result,
                                          const Dual2<Vec3>& p) const
    {
        result.set(v(), v(), v());
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<Vec3>& result,
                                          const Dual2<Vec3>& p,
                                          const Dual2<float>& t) const
    {
        result.set(v(), v(), v());
    }
    OSL_HOSTDEVICE inline Vec3 v() const { return Vec3(0.0f, 0.0f, 0.0f); };
};

struct UNullNoise {
    OSL_HOSTDEVICE UNullNoise() {}
    OSL_HOSTDEVICE inline void operator()(float& result, float x) const
    {
        result = 0.5f;
    }
    OSL_HOSTDEVICE inline void operator()(float& result, float x, float y) const
    {
        result = 0.5f;
    }
    OSL_HOSTDEVICE inline void operator()(float& result, const Vec3& p) const
    {
        result = 0.5f;
    }
    OSL_HOSTDEVICE inline void operator()(float& result, const Vec3& p,
                                          float t) const
    {
        result = 0.5f;
    }
    OSL_HOSTDEVICE inline void operator()(Vec3& result, float x) const
    {
        result = v();
    }
    OSL_HOSTDEVICE inline void operator()(Vec3& result, float x, float y) const
    {
        result = v();
    }
    OSL_HOSTDEVICE inline void operator()(Vec3& result, const Vec3& p) const
    {
        result = v();
    }
    OSL_HOSTDEVICE inline void operator()(Vec3& result, const Vec3& p,
                                          float t) const
    {
        result = v();
    }
    OSL_HOSTDEVICE inline void
    operator()(Dual2<float>& result, const Dual2<float>& x, int seed = 0) const
    {
        result.set(0.5f, 0.5f, 0.5f);
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<float>& result,
                                          const Dual2<float>& x,
                                          const Dual2<float>& y,
                                          int seed = 0) const
    {
        result.set(0.5f, 0.5f, 0.5f);
    }
    OSL_HOSTDEVICE inline void
    operator()(Dual2<float>& result, const Dual2<Vec3>& p, int seed = 0) const
    {
        result.set(0.5f, 0.5f, 0.5f);
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<float>& result,
                                          const Dual2<Vec3>& p,
                                          const Dual2<float>& t,
                                          int seed = 0) const
    {
        result.set(0.5f, 0.5f, 0.5f);
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<Vec3>& result,
                                          const Dual2<float>& x) const
    {
        result.set(v(), v(), v());
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<Vec3>& result,
                                          const Dual2<float>& x,
                                          const Dual2<float>& y) const
    {
        result.set(v(), v(), v());
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<Vec3>& result,
                                          const Dual2<Vec3>& p) const
    {
        result.set(v(), v(), v());
    }
    OSL_HOSTDEVICE inline void operator()(Dual2<Vec3>& result,
                                          const Dual2<Vec3>& p,
                                          const Dual2<float>& t) const
    {
        result.set(v(), v(), v());
    }
    OSL_HOSTDEVICE inline Vec3 v() const { return Vec3(0.5f, 0.5f, 0.5f); };
};

}  // namespace pvt
OSL_NAMESPACE_EXIT
