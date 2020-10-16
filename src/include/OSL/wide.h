// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage
// Contributions Copyright (c) 2017 Intel Inc., et al.

// clang-format off

#pragma once

#include <type_traits>

#include <OSL/oslconfig.h>
#include <OSL/dual_vec.h>
#include <OSL/Imathx/Imathx.h>

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
// A Block should not be passed around, instead Wide<DataT, WidthT>
// will hold a reference to a Block and provide access to its data.
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

template <typename DataT, int WidthT>
struct Wide;
// Reference to Block that provides a proxy to access to DataT
// for an individual data lane inside the Block.
// Respects const correctness DataT, ie: Wide<const float, 16>.
// Handles DataT being fixed size array [7], Wide: wide<const float[7], 16>
// Handles DataT being unbounded array [], Wide: wide<cpmst float[], 16>
// Implementations should support the following interface:
//{
//    static constexpr int width = WidthT;
//
//    impl-defined-proxy operator[](int lane);  // when DataT is not const
//    impl-defined-const-proxy operator[](int lane) const
//
//    // When DataT is ElementType[] unbounded array
//    typedef impl-defined ElementType;
//    typedef impl-defined NonConstElementType;
//    int length() const; // length of unbounded array
//
//    // Provide Wide access to individual array element
//    Wide<ElementType, WidthT> get_element(int array_index) const
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

namespace pvt {

    template<typename DataT, int WidthT, bool IsConstT>
    struct WideImpl; // undefined

    template <typename DataT, int WidthT>
    struct LaneProxy
    {
        typedef DataT const ValueType;

        explicit OSL_FORCEINLINE
        LaneProxy(Block<DataT, WidthT> & ref_wide_data, const int lane)
        : m_ref_wide_data(ref_wide_data)
        , m_lane(lane)
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        LaneProxy(const LaneProxy &other)
        : m_ref_wide_data(other.m_ref_wide_data)
        , m_lane(other.m_lane)
        {}

        OSL_FORCEINLINE
        operator ValueType () const
        {
            return m_ref_wide_data.get(m_lane);
        }

        OSL_FORCEINLINE const DataT &
        operator = (const DataT & value) const
        {
            m_ref_wide_data.set(m_lane, value);
            return value;
        }

    private:
        Block<DataT, WidthT> & m_ref_wide_data;
        const int m_lane;
    };

    template <typename ConstDataT, int WidthT>
    struct ConstLaneProxy
    {
        typedef typename std::remove_const<ConstDataT>::type DataType;
        typedef ConstDataT ValueType;

        explicit OSL_FORCEINLINE
        ConstLaneProxy(const Block<DataType, WidthT> & ref_wide_data, const int lane)
        : m_ref_wide_data(ref_wide_data)
        , m_lane(lane)
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        ConstLaneProxy(const ConstLaneProxy &other)
        : m_ref_wide_data(other.m_ref_wide_data)
        , m_lane(other.m_lane)
        {}

        OSL_FORCEINLINE
        operator ValueType () const
        {
            return m_ref_wide_data.get(m_lane);
        }

    private:
        const Block<DataType, WidthT> & m_ref_wide_data;
        const int m_lane;
    };

    template <typename ConstDataT, int ArrayLenT, int WidthT>
    struct ConstWideArrayLaneProxy
    {
        typedef typename std::remove_const<ConstDataT>::type DataType;

        explicit OSL_FORCEINLINE
        ConstWideArrayLaneProxy(const Block<DataType, WidthT> * array_of_wide_data, int lane)
        : m_array_of_wide_data(array_of_wide_data)
        , m_lane(lane)
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        ConstWideArrayLaneProxy(const ConstWideArrayLaneProxy &other)
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_lane(other.m_lane)
        {}

        OSL_FORCEINLINE int
        length() const { return ArrayLenT; }

        OSL_FORCEINLINE ConstLaneProxy<ConstDataT, WidthT>
        operator[](int array_index) const
        {
            OSL_DASSERT(array_index < ArrayLenT);
            return ConstLaneProxy<ConstDataT, WidthT>(m_array_of_wide_data[array_index], m_lane);
        }

    private:
        const Block<DataType, WidthT> * m_array_of_wide_data;
        const int m_lane;
    };

    template <typename ConstDataT, int WidthT>
    struct ConstWideUnboundedArrayLaneProxy
    {
        typedef typename std::remove_const<ConstDataT>::type DataType;

        explicit OSL_FORCEINLINE
        ConstWideUnboundedArrayLaneProxy(const Block<DataType, WidthT> * array_of_wide_data, int array_length, int lane)
        : m_array_of_wide_data(array_of_wide_data)
        , m_array_length(array_length)
        , m_lane(lane)
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        ConstWideUnboundedArrayLaneProxy(const ConstWideUnboundedArrayLaneProxy &other)
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
        , m_lane(other.m_lane)
        {}

        OSL_FORCEINLINE int
        length() const { return m_array_length; }

        OSL_FORCEINLINE ConstLaneProxy<ConstDataT, WidthT>
        operator[](int array_index) const
        {
            OSL_DASSERT(array_index < m_array_length);
            return ConstLaneProxy<ConstDataT, WidthT>(m_array_of_wide_data[array_index], m_lane);
        }

    private:
        const Block<DataType, WidthT> * m_array_of_wide_data;
        int m_array_length;
        const int m_lane;
    };

    template <typename ConstDataT, int WidthT>
    struct ConstWideDual2UnboundedArrayLaneProxy
    {
        typedef typename std::remove_const<ConstDataT>::type DataType;
        explicit OSL_FORCEINLINE
        ConstWideDual2UnboundedArrayLaneProxy(const Block<DataType, WidthT> * array_of_wide_data, int array_length, int lane_index)
        : m_array_of_wide_data(array_of_wide_data)
        , m_array_length(array_length)
        , m_lane_index(lane_index)
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        ConstWideDual2UnboundedArrayLaneProxy(const ConstWideDual2UnboundedArrayLaneProxy &other)
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
        , m_lane_index(other.m_lane_index)
        {}

        OSL_FORCEINLINE int
        length() const { return m_array_length; }

        struct ElementProxy
        {
            typedef typename std::remove_const<ConstDataT>::type DataType;
            typedef Dual2<DataType> const ValueType;

            explicit OSL_FORCEINLINE
            ElementProxy(const Block<DataType, WidthT> *array_of_wide_data, const int lane_index, const int array_index, const int array_length)
            : m_array_of_wide_data(array_of_wide_data)
            , m_array_index(array_index)
            , m_lane_index(lane_index)
            , m_array_length(array_length)
            {}

            // Must provide user defined copy constructor to
            // get compiler to be able to follow individual
            // data members through back to original object
            // when fully inlined the proxy should disappear
            OSL_FORCEINLINE
            ElementProxy(const ElementProxy &other)
            : m_array_of_wide_data(other.m_array_of_wide_data)
            , m_array_index(other.m_array_index)
            , m_lane_index(other.m_lane_index)
            , m_array_length(other.m_array_length)
            {}

            OSL_FORCEINLINE
            operator ValueType () const
            {
                // Intentionally have local variables as an intermediate between the array accesses
                // and the constructor of the return type.  As most constructors accept a const reference
                // this can cause the array access itself to be forwarded through inlining to the constructor
                // and at a minimum loose alignment tracking, but could cause other issues.
                DataType val = m_array_of_wide_data[m_array_index].get(m_lane_index);
                DataType dx = (m_array_of_wide_data+m_array_length)[m_array_index].get(m_lane_index);
                DataType dy = (m_array_of_wide_data + 2*m_array_length)[m_array_index].get(m_lane_index);
                return Dual2<DataType> (val, dx, dy);
            }

        private:
            const Block<DataType, WidthT> * m_array_of_wide_data;
            const int m_array_index;
            const int m_lane_index;
            const int m_array_length;
        };

        OSL_FORCEINLINE ElementProxy
        operator[](int array_index) const
        {
            OSL_DASSERT(array_index < m_array_length);
            return ElementProxy(m_array_of_wide_data, m_lane_index, array_index, m_array_length);
        }

    private:
        const Block<DataType, WidthT> * m_array_of_wide_data;
        int m_array_length;
        const int m_lane_index;
    };


    template<typename DataT, int WidthT>
    OSL_NODISCARD
    Block<DataT, WidthT>* assume_aligned(Block<DataT, WidthT>* block_ptr)
    {
        static_assert(std::alignment_of<Block<DataT, WidthT>>::value == std::alignment_of<VecReg<WidthT>>::value, "Unexepected alignment");
        return assume_aligned<VecReg<WidthT>::alignment>(block_ptr);
    }

    template<typename DataT, int WidthT>
    OSL_NODISCARD
    const Block<DataT, WidthT>* assume_aligned(const Block<DataT, WidthT>* block_ptr)
    {
        static_assert(std::alignment_of<Block<DataT, WidthT>>::value == std::alignment_of<VecReg<WidthT>>::value, "Unexepected alignment");
        return assume_aligned<VecReg<WidthT>::alignment>(block_ptr);
    }

    template <typename DataT, int WidthT>
    Block<DataT, WidthT> * block_cast(void *ptr_wide_data, int derivIndex = 0)
    {
        Block<DataT, WidthT> * block_ptr = &(reinterpret_cast<Block<DataT, WidthT> *>(ptr_wide_data)[derivIndex]);
        return assume_aligned(block_ptr);
    }

    template <typename DataT, int WidthT>
    const Block<DataT, WidthT> * block_cast(const void *ptr_wide_data)
    {
        const Block<DataT, WidthT> * block_ptr = reinterpret_cast<const Block<DataT, WidthT> *>(ptr_wide_data);
        return assume_aligned(block_ptr);
    }

    template <typename DataT, int WidthT>
    OSL_FORCEINLINE const Block<DataT, WidthT> & align_block_ref(const Block<DataT, WidthT> &ref) {
        return *assume_aligned(&ref);
    }

    template <typename DataT, int WidthT>
    OSL_FORCEINLINE Block<DataT, WidthT> & align_block_ref(Block<DataT, WidthT> &ref) {
        return *assume_aligned(&ref);
    }



    template <typename DataT, int WidthT>
    struct WideImpl<DataT, WidthT, false /*IsConstT */>
    {
        static_assert(std::is_const<DataT>::value == false, "Logic Bug:  Only meant for non-const DataT, const is meant to use specialized WideImpl");
        static_assert(std::is_array<DataT>::value == false, "Logic Bug:  Only meant for non-array DataT, arrays are meant to use specialized WideImpl");
        static constexpr int width = WidthT;
        typedef DataT ValueType;

        explicit OSL_FORCEINLINE
        WideImpl(void *ptr_wide_data, int derivIndex=0)
        : m_ref_wide_data(block_cast<DataT, WidthT>(ptr_wide_data)[derivIndex])
        {}

        // Allow implicit construction
        OSL_FORCEINLINE
        WideImpl(Block<DataT, WidthT> & ref_wide_data)
        : m_ref_wide_data(align_block_ref(ref_wide_data))
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        WideImpl(const WideImpl &other) noexcept
        : m_ref_wide_data(other.m_ref_wide_data)
        {}


        OSL_FORCEINLINE Block<DataT, WidthT> &  data() const { return m_ref_wide_data; }

        typedef LaneProxy<DataT, WidthT> Proxy;
        //typedef ConstLaneProxy<DataT, WidthT> ConstProxy;

        OSL_FORCEINLINE Proxy
        operator[](int lane) const
        {
            return Proxy(m_ref_wide_data, lane);
        }

    private:
        Block<DataT, WidthT> & m_ref_wide_data;
    };

    template <typename ConstDataT, int WidthT>
    struct WideImpl<ConstDataT, WidthT, true /*IsConstT */>
    {
        static_assert(std::is_array<ConstDataT>::value == false, "Only meant for non-array ConstDataT, arrays are meant to use specialized WideImpl");

        static constexpr int width = WidthT;

        typedef ConstDataT ValueType;
        static_assert(std::is_const<ConstDataT>::value, "unexpected compiler behavior");
        typedef typename std::remove_const<ConstDataT>::type DataT;
        typedef DataT NonConstValueType;

        explicit OSL_FORCEINLINE
        WideImpl(const void *ptr_wide_data, int derivIndex=0)
        : m_ref_wide_data(block_cast<DataT, WidthT>(ptr_wide_data)[derivIndex])
        {}

        // Allow implicit construction
        OSL_FORCEINLINE
        WideImpl(const Block<DataT, WidthT> & ref_wide_data)
        : m_ref_wide_data(align_block_ref(ref_wide_data))
        {}

        // Allow implicit conversion of const Wide from non-const Wide
        OSL_FORCEINLINE
        WideImpl(const WideImpl<DataT, WidthT, false /*IsConstT */> &other)
        : m_ref_wide_data(other.m_ref_wide_data)
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        WideImpl(const WideImpl &other) noexcept
        : m_ref_wide_data(other.m_ref_wide_data)
        {}


        typedef ConstLaneProxy<ConstDataT, WidthT> ConstProxy;

        OSL_FORCEINLINE const Block<DataT, WidthT> &  data() const { return m_ref_wide_data; }

        OSL_FORCEINLINE ConstProxy const
        operator[](int lane) const
        {
            return ConstProxy(m_ref_wide_data, lane);
        }

    private:
        const Block<DataT, WidthT> & m_ref_wide_data;
    };

    template <typename ElementT, int ArrayLenT, int WidthT>
    struct WideImpl<const ElementT[ArrayLenT], WidthT, true /*IsConstT */>
    {
        static constexpr int width = WidthT;
        static constexpr int ArrayLen = ArrayLenT;
        static_assert(ArrayLen > 0, "OSL logic bug");
        typedef const ElementT ElementType;
        static_assert(std::is_const<ElementType>::value, "unexpected compiler behavior");
        typedef ElementT NonConstElementType;
        typedef ElementType ArrayType[ArrayLen];

        explicit OSL_FORCEINLINE
        WideImpl(const void *ptr_wide_data)
        : m_array_of_wide_data(block_cast<ElementT, WidthT>(ptr_wide_data))
        {}

        explicit OSL_FORCEINLINE
        WideImpl(const Block<ElementT, WidthT> *array_of_wide_data)
        : m_array_of_wide_data(assume_aligned(array_of_wide_data))
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        WideImpl(const WideImpl &other) noexcept
        : m_array_of_wide_data(other.m_array_of_wide_data)
        {}

        static constexpr OSL_FORCEINLINE int
        length() { return ArrayLen; }


        typedef ConstWideArrayLaneProxy<ElementType, ArrayLen, WidthT> Proxy;

        OSL_FORCEINLINE Proxy const
        operator[](int lane) const
        {
            return Proxy(m_array_of_wide_data, lane);
        }

        OSL_FORCEINLINE Wide<ElementType, WidthT>
        get_element(int array_index) const
        {
            OSL_DASSERT(array_index < ArrayLen);
            return Wide<ElementType, WidthT>(m_array_of_wide_data[array_index]);
        }

    private:
        const Block<ElementT, WidthT> * m_array_of_wide_data;
    };

    template <typename ElementT, int WidthT>
    struct WideImpl<const ElementT[], WidthT, true /*IsConstT */>
    {
        static constexpr int width = WidthT;
        typedef const ElementT ElementType;
        static_assert(std::is_const<ElementType>::value, "unexpected compiler behavior");
        typedef ElementT NonConstElementType;

        explicit OSL_FORCEINLINE
        WideImpl(const void *ptr_wide_data, int array_length)
        : m_array_of_wide_data(block_cast<ElementT, WidthT>(ptr_wide_data))
        , m_array_length(array_length)
        {}

        explicit OSL_FORCEINLINE
        WideImpl(const Block<ElementT, WidthT> *array_of_wide_data, int array_length)
        : m_array_of_wide_data(assume_aligned(array_of_wide_data))
        , m_array_length(array_length)
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        WideImpl(const WideImpl &other) noexcept
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
        {}

        OSL_FORCEINLINE int
        length() { return m_array_length; }


        typedef ConstWideUnboundedArrayLaneProxy<ElementType, WidthT> Proxy;

        OSL_FORCEINLINE Proxy const
        operator[](int lane) const
        {
            return Proxy(m_array_of_wide_data, m_array_length, lane);
        }

        OSL_FORCEINLINE Wide<ElementType, WidthT>
        get_element(int array_index) const
        {
            OSL_DASSERT(array_index < m_array_length);
            return Wide<ElementType, WidthT>(m_array_of_wide_data[array_index]);
        }

    private:
        const Block<ElementT, WidthT> * m_array_of_wide_data;
        int m_array_length;
    };

    template <typename ElementT, int WidthT>
    struct WideImpl<const Dual2<ElementT>[], WidthT, true /*IsConstT */>
    {
        static constexpr int width = WidthT;
        typedef const Dual2<ElementT> ElementType;
        typedef Dual2<ElementT> NonConstElementType;

        explicit OSL_FORCEINLINE
        WideImpl(const void *ptr_wide_data, int array_length)
        : m_array_of_wide_data(block_cast<ElementT, WidthT>(ptr_wide_data))
        , m_array_length(array_length)
        {}

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        WideImpl(const WideImpl &other) noexcept
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
        {}

        OSL_FORCEINLINE int
        length() { return m_array_length; }

        typedef ConstWideDual2UnboundedArrayLaneProxy<ElementT, WidthT> Proxy;

        OSL_FORCEINLINE Proxy const
        operator[](int lane_index) const
        {
            return Proxy(m_array_of_wide_data, m_array_length, lane_index);
        }

        // get_element doesn't work here as val, dx, dy will be separated
        // in memory by m_array_length.  Perhaps could add get_val(), get_dx(), get_dy()
        // similar to MaskedData in order to enable get_element.
    private:
        const Block<ElementT, WidthT> * m_array_of_wide_data;
        int m_array_length;
    };

} // namespace pvt


#if OSL_INTEL_COMPILER || OSL_GNUC_VERSION
    // Workaround for error #3466: inheriting constructors must be inherited from a direct base class
    #define __OSL_INHERIT_BASE_CTORS(DERIVED, BASE) \
        using Base = typename DERIVED::BASE; \
        using Base::BASE;
#else
    #define __OSL_INHERIT_BASE_CTORS(DERIVED, BASE) \
        using Base = typename DERIVED::BASE; \
        using Base::Base;
#endif

template <typename DataT, int WidthT>
struct Wide
: pvt::WideImpl<DataT, WidthT, std::is_const<DataT>::value>
{
    __OSL_INHERIT_BASE_CTORS(Wide,WideImpl)
    static constexpr int width = WidthT;
};


#undef __OSL_INHERIT_BASE_CTORS

OSL_NAMESPACE_EXIT
