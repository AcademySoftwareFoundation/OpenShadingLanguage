/*
Copyright (c) 2017 Intel Inc., et al.
All Rights Reserved.
 
Copyright (c) 2009-2013 Sony Pictures Imageworks Inc., et al.
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

#pragma once

#include <type_traits>

#include <OSL/oslconfig.h>
//#include "mask.h"
#include "dual_vec.h"
#include "Imathx.h"

OSL_NAMESPACE_ENTER

// template<DataT>
// struct ImplDefinedProxy
//{
//    typedef DataT ValueType;
//    operator DataT () const;
//    const DataT & operator = (const DataT & value) const;
//};
//
// A Proxy object abstracts the location and layout of some DataT.
// DataT can be imported, with an assignment
// operator, or exported, with a conversion operator.
// Direct pointer access is not provided as the
// data layout inside may not adhere to C ABI.
// This approach enables user code to use DataT
// and let Proxy objects handle moving the data
// in-between layouts (AOS<->SOA).
// "unproxy(impl_proxy)" will extract the correctly type value.
// NOTE: assignment operator is const, making Proxy objects
// suitable to be passed by value through lambda closures

// Exporting data out of a proxy requires a conversion operator,
// which requires the left hand side of an expression to be correctly
// typed (because its a parameter or static_cast).
// Correctly typed usage may not be present in the users code,
//
// IE:  std::cout << proxy_obj;
//
// and cause a compilation failure.
// To work around this, a helper free function is provided to export the
// correctly typed value of a proxy object
//
// typename ImplDefinedProxy::ValueType const unproxy(const ImplDefinedProxy &proxy);
//
// IE:  std::cout << unproxy(proxy_obj);


template <typename DataT, int WidthT>
struct Block;
// A Block provides physical storage for WidthT entries of DataT,
// WidthT is typically set to the # of physical SIMD data lanes
// on a system.
// The data itself is stored in a SOA (Structure of Arrays) layout.
// DataT may be Dual2<T>.
// DataT must NOT be an array, arrays are supported by having
// and array of Block[].
// Implementations should support the following interface:
//{
//    Block() = default;
//    // We want to avoid accidentally copying these when the intent was to just pass a reference,
//    // especially with lambda closures
//    Block(const Block &other) = delete;
//    // Use default constructor + assignment operator to effectively copy construct
//
//    template<typename... DataListT, typename = pvt::enable_if_type<(sizeof...(DataListT) == WidthT)> >
//    explicit OSL_FORCEINLINE
//    Block(const DataListT &...values);
//
//    void set(int lane, const DataT & value);  // when DataT is not const
//    DataT get(int lane) const;
//
//    impl-defined-proxy operator[](int lane);  // when DataT is not const
//    impl-defined-const-proxy operator[](int lane) const
//
//    void dump(const char *name) const;
//};

// More wrappers will be added here to wrap a reference to Block data along with a mask...

// Utilities to assign all data lanes to the same value
template <typename DataT, int WidthT>
OSL_FORCEINLINE void assign_all(Block<DataT, WidthT> &, const DataT &);









// IMPLEMENTATION BELOW
// NOTE: not all combinations of DataT, const DataT, DataT[], DataT[3] are implemented
// only specialization actually used by the current code base are here.
// NOTE: additional constructors & helpers functions exist in the implementation
// that were not specified in the descriptions above for brevity.

static constexpr int MaxSupportedSimdLaneCount = 16;

/// Type for an opaque pointer to whatever the renderer uses to represent a
/// coordinate transformation.
typedef const void * TransformationPtr;

namespace pvt {
    // Forward declarations
    template <typename DataT, int WidthT>
    struct LaneProxy;
    template <typename ConstDataT, int WidthT>
    struct ConstLaneProxy;
};

// Type to establish proper alignment for a vector register of a given width.
// Can be used with alignas(VecReg<WidthT>) attribute
// or be a base class to force derived class to adhere to
// its own alignment restrictions
template <int WidthT>
struct alignas(WidthT*sizeof(float)) VecReg {
    // NOTE: regardless of the actual type, our goal is to
    // establish the # of bytes a vector registor holds
    // for that purpose we just use float.
    // Should OSL::Float change to double this would need
    // to as well.
    static constexpr int alignment = WidthT*sizeof(float);
};

static_assert(std::alignment_of<VecReg<16>>::value == 64, "Unexepected alignment");
static_assert(std::alignment_of<VecReg<8>>::value == 32, "Unexepected alignment");
static_assert(std::alignment_of<VecReg<4>>::value == 16, "Unexepected alignment");
static_assert(std::alignment_of<VecReg<16>>::value == VecReg<16>::alignment, "Unexepected alignment");
static_assert(std::alignment_of<VecReg<8>>::value == VecReg<8>::alignment, "Unexepected alignment");
static_assert(std::alignment_of<VecReg<4>>::value == VecReg<4>::alignment, "Unexepected alignment");


template <typename BuiltinT, int WidthT>
struct alignas(VecReg<WidthT>) BlockOfBuiltin
{
    typedef BuiltinT ValueType;
    static constexpr int width = WidthT;

    ValueType data[WidthT];

    OSL_FORCEINLINE void
    set(int lane, ValueType value)
    {
        data[lane] = value;
    }

    OSL_FORCEINLINE void
    set(int lane, ValueType value, bool laneMask)
    {
        if (laneMask) {
            data[lane] = value;
        }
    }

    OSL_FORCEINLINE void
    set_all(ValueType value)
    {
        OSL_FORCEINLINE_BLOCK
        {
            OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
            for(int i = 0; i < WidthT; ++i)
            {
                data[i] = value;
            }
        }
    }

protected:
    template<int HeadIndexT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT>, const ValueType & value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... BuiltinListT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT, TailIndexListT...>, ValueType headValue, BuiltinListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    OSL_FORCEINLINE BlockOfBuiltin() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    BlockOfBuiltin(const BlockOfBuiltin &other) = delete;

    template<typename... BuiltinListT, typename = pvt::enable_if_type<(sizeof...(BuiltinListT) == WidthT)> >
    explicit OSL_FORCEINLINE
    BlockOfBuiltin(const BuiltinListT &...values)
    {
        typedef pvt::make_int_sequence<sizeof...(BuiltinListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }

    OSL_FORCEINLINE BuiltinT
    get(int lane) const
    {
        return data[lane];
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(static_cast<Block<ValueType, WidthT> &>(*this), lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(static_cast<const Block<ValueType, WidthT> &>(*this), lane);
    }

    void dump(const char *name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << data[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
    }
};





// Specializations
template <int WidthT>
struct Block<float, WidthT> : public BlockOfBuiltin<float, WidthT> {};

template <int WidthT>
struct Block<int, WidthT> : public BlockOfBuiltin<int, WidthT> {};

template <int WidthT>
struct Block<TransformationPtr, WidthT> : public BlockOfBuiltin<TransformationPtr, WidthT> {};


// Vec4 isn't used by external interfaces, but some internal
// noise functions utilize a wide version of it.
typedef Imath::Vec4<Float>     Vec4;

template <int WidthT>
struct alignas(VecReg<WidthT>) Block<Vec4, WidthT>
{
    typedef Vec4 ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];
    float z[WidthT];
    float w[WidthT];

    OSL_FORCEINLINE void
    set(int lane, const Vec4 & value)
    {
        x[lane] = value.x;
        y[lane] = value.y;
        z[lane] = value.z;
        w[lane] = value.w;
    }

    OSL_FORCEINLINE void
    set(int lane, const Vec4 & value, bool laneMask)
    {
        // Encourage blend operation with per
        // component test of mask
        if (laneMask)
            x[lane] = value.x;
        if (laneMask)
            y[lane] = value.y;
        if (laneMask)
            z[lane] = value.z;
        if (laneMask)
            w[lane] = value.w;
    }


protected:
    template<int HeadIndexT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT>, const Vec4 & value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Vec4ListT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT, TailIndexListT...>, Vec4 headValue, Vec4ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block &other) = delete;

    template<typename... Vec4ListT, typename = pvt::enable_if_type<(sizeof...(Vec4ListT) == WidthT)> >
    explicit OSL_FORCEINLINE
    Block(const Vec4ListT &...values)
    {
        typedef pvt::make_int_sequence<sizeof...(Vec4ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE Vec4
    get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        float lx = x[lane];
        float ly = y[lane];
        float lz = z[lane];
        float lw = w[lane];

        return Vec4(lx, ly, lz, lw);

    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }

    void dump(const char *name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "x{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << x[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "y{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << y[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "z{"    ;
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << z[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "w{"    ;
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << w[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
    }

};

template <int WidthT>
struct alignas(VecReg<WidthT>) Block<Vec3, WidthT>
{
    typedef Vec3 ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];
    float z[WidthT];

    OSL_FORCEINLINE void
    set(int lane, const Vec3 & value)
    {
        x[lane] = value.x;
        y[lane] = value.y;
        z[lane] = value.z;
    }

    OSL_FORCEINLINE void
    set(int lane, const Vec3 & value, bool laneMask)
    {
        // Encourage blend operation with per
        // component test of mask
        if (laneMask)
            x[lane] = value.x;
        if (laneMask)
            y[lane] = value.y;
        if (laneMask)
            z[lane] = value.z;
    }

protected:
    template<int HeadIndexT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT>, const Vec3 & value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Vec3ListT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT, TailIndexListT...>, Vec3 headValue, Vec3ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block &other) = delete;

    template<typename... Vec3ListT, typename = pvt::enable_if_type<(sizeof...(Vec3ListT) == WidthT)> >
    explicit OSL_FORCEINLINE
    Block(const Vec3ListT &...values)
    {
        typedef pvt::make_int_sequence<sizeof...(Vec3ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE Vec3
    get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        float lx = x[lane];
        float ly = y[lane];
        float lz = z[lane];

        return Vec3(lx, ly, lz);
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }

    void dump(const char *name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "x{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << x[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "y{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << y[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "z{"    ;
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << z[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
    }

};

template <int WidthT>
struct alignas(VecReg<WidthT>) Block<Vec2, WidthT>
{
    typedef Vec2 ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];

    OSL_FORCEINLINE void
    set(int lane, const Vec2 & value)
    {
        x[lane] = value.x;
        y[lane] = value.y;
    }

    OSL_FORCEINLINE void
    set(int lane, const Vec2 & value, bool laneMask)
    {
        // Encourage blend operation with per
        // component test of mask
        if (laneMask)
            x[lane] = value.x;
        if (laneMask)
            y[lane] = value.y;
    }

protected:
    template<int HeadIndexT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT>, const Vec2 & value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Vec2ListT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT, TailIndexListT...>, Vec2 headValue, Vec2ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block &other) = delete;

    template<typename... Vec2ListT, typename = pvt::enable_if_type<(sizeof...(Vec2ListT) == WidthT)> >
    explicit OSL_FORCEINLINE
    Block(const Vec2ListT &...values)
    {
        typedef pvt::make_int_sequence<sizeof...(Vec2ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE Vec2
    get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        float lx = x[lane];
        float ly = y[lane];

        return Vec2(lx, ly);
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }

    void dump(const char *name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "x{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << x[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "y{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << y[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
    }

};

template <int WidthT>
struct alignas(VecReg<WidthT>) Block<Color3, WidthT>
{
    typedef Color3 ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];
    float z[WidthT];

    OSL_FORCEINLINE void
    set(int lane, const Color3 & value)
    {
        x[lane] = value.x;
        y[lane] = value.y;
        z[lane] = value.z;
    }

    OSL_FORCEINLINE void
    set(int lane, const Color3 & value, bool laneMask)
    {
        // Encourage blend operation with per
        // component test of mask
        if (laneMask)
            x[lane] = value.x;
        if (laneMask)
            y[lane] = value.y;
        if (laneMask)
            z[lane] = value.z;
    }

protected:
    template<int HeadIndexT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT>, const Color3 & value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Color3ListT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT, TailIndexListT...>, Color3 headValue, Color3ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block &other) = delete;

    template<typename... Color3ListT, typename = pvt::enable_if_type<(sizeof...(Color3ListT) == WidthT)> >
    explicit OSL_FORCEINLINE
    Block(const Color3ListT &...values)
    {
        typedef pvt::make_int_sequence<sizeof...(Color3ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE Color3
    get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        float lx = x[lane];
        float ly = y[lane];
        float lz = z[lane];

        return Color3(lx, ly, lz);
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }

    void dump(const char *name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "x{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << x[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "y{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << y[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "z{"    ;
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << z[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
    }

};

template <int WidthT>
struct alignas(VecReg<WidthT>) Block<Matrix44, WidthT>
{
    typedef Matrix44 ValueType;
    static constexpr int width = WidthT;

    float x00[WidthT];
    float x01[WidthT];
    float x02[WidthT];
    float x03[WidthT];

    float x10[WidthT];
    float x11[WidthT];
    float x12[WidthT];
    float x13[WidthT];

    float x20[WidthT];
    float x21[WidthT];
    float x22[WidthT];
    float x23[WidthT];

    float x30[WidthT];
    float x31[WidthT];
    float x32[WidthT];
    float x33[WidthT];

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block &other) = delete;

    OSL_FORCEINLINE void
    set(int lane, const Matrix44 & value)
    {
        x00[lane] = value.x[0][0];
        x01[lane] = value.x[0][1];
        x02[lane] = value.x[0][2];
        x03[lane] = value.x[0][3];

        x10[lane] = value.x[1][0];
        x11[lane] = value.x[1][1];
        x12[lane] = value.x[1][2];
        x13[lane] = value.x[1][3];

        x20[lane] = value.x[2][0];
        x21[lane] = value.x[2][1];
        x22[lane] = value.x[2][2];
        x23[lane] = value.x[2][3];

        x30[lane] = value.x[3][0];
        x31[lane] = value.x[3][1];
        x32[lane] = value.x[3][2];
        x33[lane] = value.x[3][3];
    }

    OSL_FORCEINLINE void
    set(int lane, const Matrix44 & value, bool laneMask)
    {
        // Encourage blend operation with per
        // component test of mask
        if (laneMask)
            x00[lane] = value.x[0][0];
        if (laneMask)
            x01[lane] = value.x[0][1];
        if (laneMask)
            x02[lane] = value.x[0][2];
        if (laneMask)
            x03[lane] = value.x[0][3];

        if (laneMask)
            x10[lane] = value.x[1][0];
        if (laneMask)
            x11[lane] = value.x[1][1];
        if (laneMask)
            x12[lane] = value.x[1][2];
        if (laneMask)
            x13[lane] = value.x[1][3];

        if (laneMask)
            x20[lane] = value.x[2][0];
        if (laneMask)
            x21[lane] = value.x[2][1];
        if (laneMask)
            x22[lane] = value.x[2][2];
        if (laneMask)
            x23[lane] = value.x[2][3];

        if (laneMask)
            x30[lane] = value.x[3][0];
        if (laneMask)
            x31[lane] = value.x[3][1];
        if (laneMask)
            x32[lane] = value.x[3][2];
        if (laneMask)
            x33[lane] = value.x[3][3];
    }

    OSL_FORCEINLINE Matrix44
    get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        float v00 = x00[lane];
        float v01 = x01[lane];
        float v02 = x02[lane];
        float v03 = x03[lane];

        float v10 = x10[lane];
        float v11 = x11[lane];
        float v12 = x12[lane];
        float v13 = x13[lane];

        float v20 = x20[lane];
        float v21 = x21[lane];
        float v22 = x22[lane];
        float v23 = x23[lane];

        float v30 = x30[lane];
        float v31 = x31[lane];
        float v32 = x32[lane];
        float v33 = x33[lane];

        return Matrix44(
            v00, v01, v02, v03,
            v10, v11, v12, v13,
            v20, v21, v22, v23,
            v30, v31, v32, v33
            );
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};

template <int WidthT>
struct alignas(VecReg<WidthT>) Block<ustring, WidthT>
{
    static constexpr int width = WidthT;
    typedef ustring ValueType;

    // To enable vectorization, use uintptr_t to store the ustring (const char *)
    uintptr_t str[WidthT];
    static_assert(sizeof(ustring) == sizeof(const char*), "ustring must be pointer size");

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block &other) = delete;

    OSL_FORCEINLINE void
    set(int lane, const ustring& value)
    {
        str[lane] = reinterpret_cast<uintptr_t>(value.c_str());
    }

    OSL_FORCEINLINE void
    set(int lane, const ustring& value, bool laneMask)
    {
        if (laneMask)
            str[lane] = reinterpret_cast<uintptr_t>(value.c_str());
    }

    OSL_FORCEINLINE ustring
    get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        auto unique_cstr = reinterpret_cast<const char *>(str[lane]);
        return ustring::from_unique(unique_cstr);
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};

template <int WidthT>
struct alignas(VecReg<WidthT>) Block<Dual2<float>, WidthT>
{
    typedef Dual2<float> ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float dx[WidthT];
    float dy[WidthT];

    OSL_FORCEINLINE void
    set(int lane, const ValueType & value)
    {
        x[lane] = value.val();
        dx[lane] = value.dx();
        dy[lane] = value.dy();
    }

    OSL_FORCEINLINE void
    set(int lane, const ValueType & value, bool laneMask)
    {
        // Encourage blend operation with per
        // component test of mask
        if (laneMask)
            x[lane] = value.val();
        if (laneMask)
            dx[lane] = value.dx();
        if (laneMask)
            dy[lane] = value.dy();
    }


protected:
    template<int HeadIndexT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT>, const ValueType &value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT, TailIndexListT...>, ValueType headValue, ValueListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block &other) = delete;

    template<typename... ValueListT, typename = pvt::enable_if_type<(sizeof...(ValueListT) == WidthT)> >
    explicit OSL_FORCEINLINE
    Block(const ValueListT &...values)
    {
        typedef pvt::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE ValueType
    get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        float lx = x[lane];
        float ldx = dx[lane];
        float ldy = dy[lane];
        return ValueType(lx, ldx, ldy);
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};

template <int WidthT>
struct alignas(VecReg<WidthT>) Block<Dual2<Vec3>, WidthT>
{
    typedef Dual2<Vec3> ValueType;
    static constexpr int width = WidthT;

    float val_x[WidthT];
    float val_y[WidthT];
    float val_z[WidthT];

    float dx_x[WidthT];
    float dx_y[WidthT];
    float dx_z[WidthT];

    float dy_x[WidthT];
    float dy_y[WidthT];
    float dy_z[WidthT];

    OSL_FORCEINLINE void
    set(int lane, const ValueType & value)
    {
        val_x[lane] = value.val().x;
        val_y[lane] = value.val().y;
        val_z[lane] = value.val().z;

        dx_x[lane] = value.dx().x;
        dx_y[lane] = value.dx().y;
        dx_z[lane] = value.dx().z;

        dy_x[lane] = value.dy().x;
        dy_y[lane] = value.dy().y;
        dy_z[lane] = value.dy().z;
    }

    OSL_FORCEINLINE void
    set(int lane, const ValueType & value, bool laneMask)
    {
        // Encourage blend operation with per
        // component test of mask
        if (laneMask)
            val_x[lane] = value.val().x;
        if (laneMask)
            val_y[lane] = value.val().y;
        if (laneMask)
            val_z[lane] = value.val().z;

        if (laneMask)
            dx_x[lane] = value.dx().x;
        if (laneMask)
            dx_y[lane] = value.dx().y;
        if (laneMask)
            dx_z[lane] = value.dx().z;

        if (laneMask)
            dy_x[lane] = value.dy().x;
        if (laneMask)
            dy_y[lane] = value.dy().y;
        if (laneMask)
            dy_z[lane] = value.dy().z;
    }

protected:
    template<int HeadIndexT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT>, const ValueType &value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT, TailIndexListT...>, ValueType headValue, ValueListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block &other) = delete;

    template<typename... ValueListT, typename = pvt::enable_if_type<(sizeof...(ValueListT) == WidthT)> >
    explicit OSL_FORCEINLINE
    Block(const ValueListT &...values)
    {
        typedef pvt::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE ValueType
    get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        float lval_x = val_x[lane];
        float lval_y = val_y[lane];
        float lval_z = val_z[lane];

        float ldx_x = dx_x[lane];
        float ldx_y = dx_y[lane];
        float ldx_z = dx_z[lane];

        float ldy_x = dy_x[lane];
        float ldy_y = dy_y[lane];
        float ldy_z = dy_z[lane];


        return ValueType(Vec3(lval_x, lval_y, lval_z),
                Vec3(ldx_x, ldx_y, ldx_z),
                Vec3(ldy_x, ldy_y, ldy_z));
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};



template <int WidthT>
struct alignas(VecReg<WidthT>) Block<Dual2<Color3>, WidthT>
{
    typedef Dual2<Color3> ValueType;
    static constexpr int width = WidthT;
    float val_x[WidthT];
    float val_y[WidthT];
    float val_z[WidthT];

    float dx_x[WidthT];
    float dx_y[WidthT];
    float dx_z[WidthT];

    float dy_x[WidthT];
    float dy_y[WidthT];
    float dy_z[WidthT];

    OSL_FORCEINLINE void
    set(int lane, const ValueType & value)
    {
        val_x[lane] = value.val().x;
        val_y[lane] = value.val().y;
        val_z[lane] = value.val().z;

        dx_x[lane] = value.dx().x;
        dx_y[lane] = value.dx().y;
        dx_z[lane] = value.dx().z;

        dy_x[lane] = value.dy().x;
        dy_y[lane] = value.dy().y;
        dy_z[lane] = value.dy().z;
    }

    OSL_FORCEINLINE void
    set(int lane, const ValueType & value, bool laneMask)
    {
        // Encourage blend operation with per
        // component test of mask
        if (laneMask)
            val_x[lane] = value.val().x;
        if (laneMask)
            val_y[lane] = value.val().y;
        if (laneMask)
            val_z[lane] = value.val().z;

        if (laneMask)
            dx_x[lane] = value.dx().x;
        if (laneMask)
            dx_y[lane] = value.dx().y;
        if (laneMask)
            dx_z[lane] = value.dx().z;

        if (laneMask)
            dy_x[lane] = value.dy().x;
        if (laneMask)
            dy_y[lane] = value.dy().y;
        if (laneMask)
            dy_z[lane] = value.dy().z;
    }

protected:
    template<int HeadIndexT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT>, const ValueType &value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
    OSL_FORCEINLINE void
    set(pvt::int_sequence<HeadIndexT, TailIndexListT...>, ValueType headValue, ValueListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block &other) = delete;

    template<typename... ValueListT, typename = pvt::enable_if_type<(sizeof...(ValueListT) == WidthT)> >
    explicit OSL_FORCEINLINE
    Block(const ValueListT &...values)
    {
        typedef pvt::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }

    OSL_FORCEINLINE ValueType
    get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        float lval_x = val_x[lane];
        float lval_y = val_y[lane];
        float lval_z = val_z[lane];

        float ldx_x = dx_x[lane];
        float ldx_y = dx_y[lane];
        float ldx_z = dx_z[lane];

        float ldy_x = dy_x[lane];
        float ldy_y = dy_y[lane];
        float ldy_z = dy_z[lane];


        return ValueType(Vec3(lval_x, lval_y, lval_z),
                Vec3(ldx_x, ldx_y, ldx_z),
                Vec3(ldy_x, ldy_y, ldy_z));
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT>
    operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};

template <typename DataT, int WidthT>
OSL_FORCEINLINE void
assign_all(Block<DataT, WidthT> &wide_data, const DataT &value)
{
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
        for(int i = 0; i < WidthT; ++i) {
            wide_data.set(i, value);
        }
    }
}


OSL_NAMESPACE_EXIT
