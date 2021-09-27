// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
// Contributions Copyright (c) 2017 Intel Inc., et al.

#pragma once

#include <type_traits>

#include <OSL/Imathx/Imathx.h>
#include <OSL/dual_vec.h>
#include <OSL/mask.h>
#include <OSL/oslconfig.h>

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
template<typename DataT, int WidthT> struct Block;

// Typically Block's of data aren't passed around, instead
// a Wide<DataT, WidthT> or Masked<DataT, WidthT> are passed
// by value.  These wrapper's hold onto a reference to a Block
// and provide the necessary proxies to access the underlying
// data and enforce masking.
template<typename DataT, int WidthT> struct Wide;
template<typename DataT, int WidthT> struct Masked;

// To pass a single const uniform value to an algorithm designed
// to work with Wide data.
template<typename ConstDataT, int WidthT> struct UniformAsWide;

// For variant blocks of data (where the type is unknown,
// we have a special wrapper MaskedData.  Intent is type
// specific wrapper Masked<DataT, WidthT> be used to
// access the underlying data;
template<int WidthT> class MaskedData;

// For type specific access to uniform variant data
template<typename DataT> struct Ref;

// For variant uniform data (where the type is unknown,
// we have a special wrapper RefData.  Intent is type
// specific wrapper Ref<DataT> be used to
// access the underlying data;
class RefData;

// Utilities to assign all data lanes to the same value
template<typename DataT, int WidthT>
OSL_FORCEINLINE void
assign_all(Masked<DataT, WidthT>, const DataT&);
template<typename DataT, int WidthT>
OSL_FORCEINLINE void
assign_all(Block<DataT, WidthT>&, const DataT&);

// Scalar execution of Functor for each unique value in the Wide data out
// of the data_mask, the functor must be of the form
//     (const DataT &, Mask<WidthT>)->void
// where the DataT is a unique value from the wide data,
// the mask identifies which data lanes contain that unique value.
template<typename DataT, int WidthT, typename FunctorT>
OSL_FORCEINLINE void
foreach_unique(Wide<DataT, WidthT> wdata, Mask<WidthT> data_mask, FunctorT f);


// IMPLEMENTATION BELOW
// NOTE: not all combinations of DataT, const DataT, DataT[], DataT[3] are implemented
// only specialization actually used by the current code base are here.
// NOTE: additional constructors & helpers functions exist in the implementation
// that were not specified in the descriptions above for brevity.

/// Type for an opaque pointer to whatever the renderer uses to represent a
/// coordinate transformation.
typedef const void* TransformationPtr;

namespace pvt {
// Forward declarations
template<typename DataT, int WidthT> struct LaneProxy;
template<typename ConstDataT, int WidthT> struct ConstLaneProxy;
};  // namespace pvt

// Type to establish proper alignment for a vector register of a given width.
// Can be used with alignas(VecReg<WidthT>) attribute
// or be a base class to force derived class to adhere to
// its own alignment restrictions
template<int WidthT> struct alignas(WidthT * sizeof(float)) VecReg {
    // NOTE: regardless of the actual type, our goal is to
    // establish the # of bytes a vector register holds
    // for that purpose we just use float.
    // Should OSL::Float change to double this would need
    // to as well.
    static constexpr int alignment = WidthT * sizeof(float);
};

static_assert(std::alignment_of<VecReg<16>>::value == 64,
              "Unexepected alignment");
static_assert(std::alignment_of<VecReg<8>>::value == 32,
              "Unexepected alignment");
static_assert(std::alignment_of<VecReg<4>>::value == 16,
              "Unexepected alignment");
static_assert(std::alignment_of<VecReg<16>>::value == VecReg<16>::alignment,
              "Unexepected alignment");
static_assert(std::alignment_of<VecReg<8>>::value == VecReg<8>::alignment,
              "Unexepected alignment");
static_assert(std::alignment_of<VecReg<4>>::value == VecReg<4>::alignment,
              "Unexepected alignment");


template<typename BuiltinT, int WidthT>
struct alignas(VecReg<WidthT>) BlockOfBuiltin {
    typedef BuiltinT ValueType;
    static constexpr int width = WidthT;

    ValueType data[WidthT];

    OSL_FORCEINLINE void set(int lane, ValueType value) { data[lane] = value; }

    OSL_FORCEINLINE void set(int lane, ValueType value, bool laneMask)
    {
        if (laneMask) {
            data[lane] = value;
        }
    }

    OSL_FORCEINLINE void set_all(ValueType value)
    {
        OSL_FORCEINLINE_BLOCK
        {
            OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
            for (int i = 0; i < WidthT; ++i) {
                data[i] = value;
            }
        }
    }

protected:
    template<int HeadIndexT>
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT>,
                             const ValueType& value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... BuiltinListT>
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT, TailIndexListT...>,
                             ValueType headValue, BuiltinListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }

public:
    OSL_FORCEINLINE BlockOfBuiltin() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    BlockOfBuiltin(const BlockOfBuiltin& other) = delete;

    template<typename... BuiltinListT,
             typename = pvt::enable_if_type<(sizeof...(BuiltinListT) == WidthT)>>
    explicit OSL_FORCEINLINE BlockOfBuiltin(const BuiltinListT&... values)
    {
        typedef pvt::make_int_sequence<sizeof...(BuiltinListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }

    OSL_FORCEINLINE BuiltinT get(int lane) const { return data[lane]; }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(
            static_cast<Block<ValueType, WidthT>&>(*this), lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(
            static_cast<const Block<ValueType, WidthT>&>(*this), lane);
    }

    void dump(const char* name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << data[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
    }
};



// Specializations
template<int WidthT>
struct Block<float, WidthT> : public BlockOfBuiltin<float, WidthT> {
};

template<int WidthT>
struct Block<int, WidthT> : public BlockOfBuiltin<int, WidthT> {
};

template<int WidthT>
struct Block<TransformationPtr, WidthT>
    : public BlockOfBuiltin<TransformationPtr, WidthT> {
};


// Vec4 isn't used by external interfaces, but some internal
// noise functions utilize a wide version of it.
typedef Imath::Vec4<Float> Vec4;

template<int WidthT> struct alignas(VecReg<WidthT>) Block<Vec4, WidthT> {
    typedef Vec4 ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];
    float z[WidthT];
    float w[WidthT];

    OSL_FORCEINLINE void set(int lane, const Vec4& value)
    {
        x[lane] = value.x;
        y[lane] = value.y;
        z[lane] = value.z;
        w[lane] = value.w;
    }

    OSL_FORCEINLINE void set(int lane, const Vec4& value, bool laneMask)
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
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT>, const Vec4& value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Vec4ListT>
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT, TailIndexListT...>,
                             Vec4 headValue, Vec4ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }

public:
    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block& other) = delete;

    template<typename... Vec4ListT,
             typename = pvt::enable_if_type<(sizeof...(Vec4ListT) == WidthT)>>
    explicit OSL_FORCEINLINE Block(const Vec4ListT&... values)
    {
        typedef pvt::make_int_sequence<sizeof...(Vec4ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE Vec4 get(int lane) const
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

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }

    void dump(const char* name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "x{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << x[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
        std::cout << "y{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << y[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
        std::cout << "z{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << z[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
        std::cout << "w{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << w[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
    }
};

template<int WidthT> struct alignas(VecReg<WidthT>) Block<Vec3, WidthT> {
    typedef Vec3 ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];
    float z[WidthT];

    OSL_FORCEINLINE void set(int lane, const Vec3& value)
    {
        x[lane] = value.x;
        y[lane] = value.y;
        z[lane] = value.z;
    }

    OSL_FORCEINLINE void set(int lane, const Vec3& value, bool laneMask)
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
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT>, const Vec3& value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Vec3ListT>
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT, TailIndexListT...>,
                             Vec3 headValue, Vec3ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }

public:
    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block& other) = delete;

    template<typename... Vec3ListT,
             typename = pvt::enable_if_type<(sizeof...(Vec3ListT) == WidthT)>>
    explicit OSL_FORCEINLINE Block(const Vec3ListT&... values)
    {
        typedef pvt::make_int_sequence<sizeof...(Vec3ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE Vec3 get(int lane) const
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

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }

    void dump(const char* name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "x{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << x[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
        std::cout << "y{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << y[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
        std::cout << "z{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << z[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
    }
};

template<int WidthT> struct alignas(VecReg<WidthT>) Block<Vec2, WidthT> {
    typedef Vec2 ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];

    OSL_FORCEINLINE void set(int lane, const Vec2& value)
    {
        x[lane] = value.x;
        y[lane] = value.y;
    }

    OSL_FORCEINLINE void set(int lane, const Vec2& value, bool laneMask)
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
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT>, const Vec2& value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Vec2ListT>
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT, TailIndexListT...>,
                             Vec2 headValue, Vec2ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }

public:
    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block& other) = delete;

    template<typename... Vec2ListT,
             typename = pvt::enable_if_type<(sizeof...(Vec2ListT) == WidthT)>>
    explicit OSL_FORCEINLINE Block(const Vec2ListT&... values)
    {
        typedef pvt::make_int_sequence<sizeof...(Vec2ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE Vec2 get(int lane) const
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

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }

    void dump(const char* name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "x{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << x[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
        std::cout << "y{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << y[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
    }
};

template<int WidthT> struct alignas(VecReg<WidthT>) Block<Color3, WidthT> {
    typedef Color3 ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];
    float z[WidthT];

    OSL_FORCEINLINE void set(int lane, const Color3& value)
    {
        x[lane] = value.x;
        y[lane] = value.y;
        z[lane] = value.z;
    }

    OSL_FORCEINLINE void set(int lane, const Color3& value, bool laneMask)
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
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT>, const Color3& value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Color3ListT>
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT, TailIndexListT...>,
                             Color3 headValue, Color3ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }

public:
    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block& other) = delete;

    template<typename... Color3ListT,
             typename = pvt::enable_if_type<(sizeof...(Color3ListT) == WidthT)>>
    explicit OSL_FORCEINLINE Block(const Color3ListT&... values)
    {
        typedef pvt::make_int_sequence<sizeof...(Color3ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE Color3 get(int lane) const
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

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }

    void dump(const char* name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "x{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << x[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
        std::cout << "y{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << y[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
        std::cout << "z{";
        for (int i = 0; i < WidthT; ++i) {
            std::cout << z[i];
            if (i < (WidthT - 1))
                std::cout << ",";
        }
        std::cout << "}" << std::endl;
    }
};

template<int WidthT> struct alignas(VecReg<WidthT>) Block<Matrix44, WidthT> {
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
    Block(const Block& other) = delete;

    OSL_FORCEINLINE void set(int lane, const Matrix44& value)
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

    OSL_FORCEINLINE void set(int lane, const Matrix44& value, bool laneMask)
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

    OSL_FORCEINLINE Matrix44 get(int lane) const
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

        return Matrix44(v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22,
                        v23, v30, v31, v32, v33);
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};

template<int WidthT> struct alignas(VecReg<WidthT>) Block<ustring, WidthT> {
    static constexpr int width = WidthT;
    typedef ustring ValueType;

    // To enable vectorization, use uintptr_t to store the ustring (const char *)
    uintptr_t str[WidthT];
    static_assert(sizeof(ustring) == sizeof(const char*),
                  "ustring must be pointer size");

    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block& other) = delete;

    OSL_FORCEINLINE void set(int lane, const ustring& value)
    {
        str[lane] = reinterpret_cast<uintptr_t>(value.c_str());
    }

    OSL_FORCEINLINE void set(int lane, const ustring& value, bool laneMask)
    {
        if (laneMask)
            str[lane] = reinterpret_cast<uintptr_t>(value.c_str());
    }

    OSL_FORCEINLINE ustring get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        auto unique_cstr = reinterpret_cast<const char*>(str[lane]);
        return ustring::from_unique(unique_cstr);
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};

template<int WidthT>
struct alignas(VecReg<WidthT>) Block<Dual2<float>, WidthT> {
    typedef Dual2<float> ValueType;
    static constexpr int width = WidthT;
    float x[WidthT];
    float dx[WidthT];
    float dy[WidthT];

    OSL_FORCEINLINE void set(int lane, const ValueType& value)
    {
        x[lane]  = value.val();
        dx[lane] = value.dx();
        dy[lane] = value.dy();
    }

    OSL_FORCEINLINE void set(int lane, const ValueType& value, bool laneMask)
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
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT>,
                             const ValueType& value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT, TailIndexListT...>,
                             ValueType headValue, ValueListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }

public:
    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block& other) = delete;

    template<typename... ValueListT,
             typename = pvt::enable_if_type<(sizeof...(ValueListT) == WidthT)>>
    explicit OSL_FORCEINLINE Block(const ValueListT&... values)
    {
        typedef pvt::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE ValueType get(int lane) const
    {
        // Intentionally have local variables as an intermediate between the
        // array accesses and the constructor of the return type.
        // As most constructors accept a const reference this can cause the
        // array access itself to be forwarded through inlining inside the
        // constructor and possibly further.
        float lx  = x[lane];
        float ldx = dx[lane];
        float ldy = dy[lane];
        return ValueType(lx, ldx, ldy);
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};

template<int WidthT> struct alignas(VecReg<WidthT>) Block<Dual2<Vec3>, WidthT> {
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

    OSL_FORCEINLINE void set(int lane, const ValueType& value)
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

    OSL_FORCEINLINE void set(int lane, const ValueType& value, bool laneMask)
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
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT>,
                             const ValueType& value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT, TailIndexListT...>,
                             ValueType headValue, ValueListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }

public:
    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block& other) = delete;

    template<typename... ValueListT,
             typename = pvt::enable_if_type<(sizeof...(ValueListT) == WidthT)>>
    explicit OSL_FORCEINLINE Block(const ValueListT&... values)
    {
        typedef pvt::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_FORCEINLINE ValueType get(int lane) const
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
                         Vec3(ldx_x, ldx_y, ldx_z), Vec3(ldy_x, ldy_y, ldy_z));
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};



template<int WidthT>
struct alignas(VecReg<WidthT>) Block<Dual2<Color3>, WidthT> {
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

    OSL_FORCEINLINE void set(int lane, const ValueType& value)
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

    OSL_FORCEINLINE void set(int lane, const ValueType& value, bool laneMask)
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
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT>,
                             const ValueType& value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
    OSL_FORCEINLINE void set(pvt::int_sequence<HeadIndexT, TailIndexListT...>,
                             ValueType headValue, ValueListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(pvt::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }

public:
    OSL_FORCEINLINE Block() = default;
    // We want to avoid accidentally copying these when the intent was to just pass a reference
    Block(const Block& other) = delete;

    template<typename... ValueListT,
             typename = pvt::enable_if_type<(sizeof...(ValueListT) == WidthT)>>
    explicit OSL_FORCEINLINE Block(const ValueListT&... values)
    {
        typedef pvt::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }

    OSL_FORCEINLINE ValueType get(int lane) const
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
                         Vec3(ldx_x, ldx_y, ldx_z), Vec3(ldy_x, ldy_y, ldy_z));
    }

    OSL_FORCEINLINE pvt::LaneProxy<ValueType, WidthT> operator[](int lane)
    {
        return pvt::LaneProxy<ValueType, WidthT>(*this, lane);
    }

    OSL_FORCEINLINE pvt::ConstLaneProxy<const ValueType, WidthT>
    operator[](int lane) const
    {
        return pvt::ConstLaneProxy<const ValueType, WidthT>(*this, lane);
    }
};

template<typename DataT, int WidthT>
OSL_FORCEINLINE void
assign_all(Block<DataT, WidthT>& wide_data, const DataT& value)
{
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
        for (int i = 0; i < WidthT; ++i) {
            wide_data.set(i, value);
        }
    }
}

namespace pvt {

template<typename DataT, int WidthT, bool IsConstT>
struct WideImpl;  // undefined

template<typename DataT, int WidthT> struct LaneProxy {
    typedef DataT const ValueType;

    explicit OSL_FORCEINLINE LaneProxy(Block<DataT, WidthT>& ref_wide_data,
                                       const int lane)
        : m_ref_wide_data(ref_wide_data), m_lane(lane)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    LaneProxy(const LaneProxy& other)
        : m_ref_wide_data(other.m_ref_wide_data), m_lane(other.m_lane)
    {
    }

    OSL_FORCEINLINE
    operator ValueType() const { return m_ref_wide_data.get(m_lane); }

    OSL_FORCEINLINE const DataT& operator=(const DataT& value) const
    {
        m_ref_wide_data.set(m_lane, value);
        return value;
    }

private:
    Block<DataT, WidthT>& m_ref_wide_data;
    const int m_lane;
};

template<typename ConstDataT, int WidthT> struct ConstLaneProxy {
    typedef typename std::remove_const<ConstDataT>::type DataType;
    typedef ConstDataT ValueType;

    explicit OSL_FORCEINLINE
    ConstLaneProxy(const Block<DataType, WidthT>& ref_wide_data, const int lane)
        : m_ref_wide_data(ref_wide_data), m_lane(lane)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    ConstLaneProxy(const ConstLaneProxy& other)
        : m_ref_wide_data(other.m_ref_wide_data), m_lane(other.m_lane)
    {
    }

    OSL_FORCEINLINE
    operator ValueType() const { return m_ref_wide_data.get(m_lane); }

private:
    const Block<DataType, WidthT>& m_ref_wide_data;
    const int m_lane;
};

template<typename ConstDataT, int ArrayLenT, int WidthT>
struct ConstWideArrayLaneProxy {
    typedef typename std::remove_const<ConstDataT>::type DataType;

    explicit OSL_FORCEINLINE
    ConstWideArrayLaneProxy(const Block<DataType, WidthT>* array_of_wide_data,
                            int lane)
        : m_array_of_wide_data(array_of_wide_data), m_lane(lane)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    ConstWideArrayLaneProxy(const ConstWideArrayLaneProxy& other)
        : m_array_of_wide_data(other.m_array_of_wide_data), m_lane(other.m_lane)
    {
    }

    OSL_FORCEINLINE int length() const { return ArrayLenT; }

    OSL_FORCEINLINE ConstLaneProxy<ConstDataT, WidthT>
    operator[](int array_index) const
    {
        OSL_DASSERT(array_index < ArrayLenT);
        return ConstLaneProxy<ConstDataT, WidthT>(
            m_array_of_wide_data[array_index], m_lane);
    }

private:
    const Block<DataType, WidthT>* m_array_of_wide_data;
    const int m_lane;
};

template<typename ConstDataT, int WidthT>
struct ConstWideUnboundedArrayLaneProxy {
    typedef typename std::remove_const<ConstDataT>::type DataType;

    explicit OSL_FORCEINLINE ConstWideUnboundedArrayLaneProxy(
        const Block<DataType, WidthT>* array_of_wide_data, int array_length,
        int lane)
        : m_array_of_wide_data(array_of_wide_data)
        , m_array_length(array_length)
        , m_lane(lane)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    ConstWideUnboundedArrayLaneProxy(
        const ConstWideUnboundedArrayLaneProxy& other)
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
        , m_lane(other.m_lane)
    {
    }

    OSL_FORCEINLINE int length() const { return m_array_length; }

    OSL_FORCEINLINE ConstLaneProxy<ConstDataT, WidthT>
    operator[](int array_index) const
    {
        OSL_DASSERT(array_index < m_array_length);
        return ConstLaneProxy<ConstDataT, WidthT>(
            m_array_of_wide_data[array_index], m_lane);
    }

private:
    const Block<DataType, WidthT>* m_array_of_wide_data;
    int m_array_length;
    const int m_lane;
};

template<typename ConstDataT, int WidthT>
struct ConstWideDual2UnboundedArrayLaneProxy {
    typedef typename std::remove_const<ConstDataT>::type DataType;
    explicit OSL_FORCEINLINE ConstWideDual2UnboundedArrayLaneProxy(
        const Block<DataType, WidthT>* array_of_wide_data, int array_length,
        int lane_index)
        : m_array_of_wide_data(array_of_wide_data)
        , m_array_length(array_length)
        , m_lane_index(lane_index)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    ConstWideDual2UnboundedArrayLaneProxy(
        const ConstWideDual2UnboundedArrayLaneProxy& other)
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
        , m_lane_index(other.m_lane_index)
    {
    }

    OSL_FORCEINLINE int length() const { return m_array_length; }

    struct ElementProxy {
        typedef typename std::remove_const<ConstDataT>::type DataType;
        typedef Dual2<DataType> const ValueType;

        explicit OSL_FORCEINLINE
        ElementProxy(const Block<DataType, WidthT>* array_of_wide_data,
                     const int lane_index, const int array_index,
                     const int array_length)
            : m_array_of_wide_data(array_of_wide_data)
            , m_array_index(array_index)
            , m_lane_index(lane_index)
            , m_array_length(array_length)
        {
        }

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        ElementProxy(const ElementProxy& other)
            : m_array_of_wide_data(other.m_array_of_wide_data)
            , m_array_index(other.m_array_index)
            , m_lane_index(other.m_lane_index)
            , m_array_length(other.m_array_length)
        {
        }

        OSL_FORCEINLINE
        operator ValueType() const
        {
            // Intentionally have local variables as an intermediate between the array accesses
            // and the constructor of the return type.  As most constructors accept a const reference
            // this can cause the array access itself to be forwarded through inlining to the constructor
            // and at a minimum loose alignment tracking, but could cause other issues.
            DataType val = m_array_of_wide_data[m_array_index].get(
                m_lane_index);
            DataType dx
                = (m_array_of_wide_data + m_array_length)[m_array_index].get(
                    m_lane_index);
            DataType dy
                = (m_array_of_wide_data + 2 * m_array_length)[m_array_index].get(
                    m_lane_index);
            return Dual2<DataType>(val, dx, dy);
        }

    private:
        const Block<DataType, WidthT>* m_array_of_wide_data;
        const int m_array_index;
        const int m_lane_index;
        const int m_array_length;
    };

    OSL_FORCEINLINE ElementProxy operator[](int array_index) const
    {
        OSL_DASSERT(array_index < m_array_length);
        return ElementProxy(m_array_of_wide_data, m_lane_index, array_index,
                            m_array_length);
    }

private:
    const Block<DataType, WidthT>* m_array_of_wide_data;
    int m_array_length;
    const int m_lane_index;
};


template<typename DataT, int WidthT>
OSL_NODISCARD Block<DataT, WidthT>*
assume_aligned(Block<DataT, WidthT>* block_ptr)
{
    static_assert(std::alignment_of<Block<DataT, WidthT>>::value
                      == std::alignment_of<VecReg<WidthT>>::value,
                  "Unexepected alignment");
    return assume_aligned<VecReg<WidthT>::alignment>(block_ptr);
}

template<typename DataT, int WidthT>
OSL_NODISCARD const Block<DataT, WidthT>*
assume_aligned(const Block<DataT, WidthT>* block_ptr)
{
    static_assert(std::alignment_of<Block<DataT, WidthT>>::value
                      == std::alignment_of<VecReg<WidthT>>::value,
                  "Unexepected alignment");
    return assume_aligned<VecReg<WidthT>::alignment>(block_ptr);
}

template<typename DataT, int WidthT>
Block<DataT, WidthT>*
block_cast(void* ptr_wide_data, int derivIndex = 0)
{
    Block<DataT, WidthT>* block_ptr = &(
        reinterpret_cast<Block<DataT, WidthT>*>(ptr_wide_data)[derivIndex]);
    return assume_aligned(block_ptr);
}

template<typename DataT, int WidthT>
const Block<DataT, WidthT>*
block_cast(const void* ptr_wide_data)
{
    const Block<DataT, WidthT>* block_ptr
        = reinterpret_cast<const Block<DataT, WidthT>*>(ptr_wide_data);
    return assume_aligned(block_ptr);
}

template<typename DataT, int WidthT>
OSL_FORCEINLINE const Block<DataT, WidthT>&
align_block_ref(const Block<DataT, WidthT>& ref)
{
    return *assume_aligned(&ref);
}

template<typename DataT, int WidthT>
OSL_FORCEINLINE Block<DataT, WidthT>&
align_block_ref(Block<DataT, WidthT>& ref)
{
    return *assume_aligned(&ref);
}



template<typename DataT, int WidthT>
struct WideImpl<DataT, WidthT, false /*IsConstT */> {
    static_assert(
        std::is_const<DataT>::value == false,
        "Logic Bug:  Only meant for non-const DataT, const is meant to use specialized WideImpl");
    static_assert(
        std::is_array<DataT>::value == false,
        "Logic Bug:  Only meant for non-array DataT, arrays are meant to use specialized WideImpl");
    static constexpr int width = WidthT;
    typedef DataT ValueType;

    explicit OSL_FORCEINLINE WideImpl(void* ptr_wide_data, int derivIndex = 0)
        : m_ref_wide_data(block_cast<DataT, WidthT>(ptr_wide_data)[derivIndex])
    {
    }

    // Allow implicit construction
    OSL_FORCEINLINE
    WideImpl(Block<DataT, WidthT>& ref_wide_data)
        : m_ref_wide_data(align_block_ref(ref_wide_data))
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    WideImpl(const WideImpl& other) noexcept
        : m_ref_wide_data(other.m_ref_wide_data)
    {
    }


    OSL_FORCEINLINE Block<DataT, WidthT>& data() const
    {
        return m_ref_wide_data;
    }

    typedef LaneProxy<DataT, WidthT> Proxy;
    //typedef ConstLaneProxy<DataT, WidthT> ConstProxy;

    OSL_FORCEINLINE Proxy operator[](int lane) const
    {
        return Proxy(m_ref_wide_data, lane);
    }

private:
    Block<DataT, WidthT>& m_ref_wide_data;
};

template<typename ConstDataT, int WidthT>
struct WideImpl<ConstDataT, WidthT, true /*IsConstT */> {
    static_assert(
        std::is_array<ConstDataT>::value == false,
        "Only meant for non-array ConstDataT, arrays are meant to use specialized WideImpl");

    static constexpr int width = WidthT;

    typedef ConstDataT ValueType;
    static_assert(std::is_const<ConstDataT>::value,
                  "unexpected compiler behavior");
    typedef typename std::remove_const<ConstDataT>::type DataT;
    typedef DataT NonConstValueType;

    explicit OSL_FORCEINLINE WideImpl(const void* ptr_wide_data,
                                      int derivIndex = 0)
        : m_ref_wide_data(block_cast<DataT, WidthT>(ptr_wide_data)[derivIndex])
    {
    }

    // Allow implicit construction
    OSL_FORCEINLINE
    WideImpl(const Block<DataT, WidthT>& ref_wide_data)
        : m_ref_wide_data(align_block_ref(ref_wide_data))
    {
    }

    // Allow implicit conversion of const Wide from non-const Wide
    OSL_FORCEINLINE
    WideImpl(const WideImpl<DataT, WidthT, false /*IsConstT */>& other)
        : m_ref_wide_data(other.m_ref_wide_data)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    WideImpl(const WideImpl& other) noexcept
        : m_ref_wide_data(other.m_ref_wide_data)
    {
    }


    typedef ConstLaneProxy<ConstDataT, WidthT> ConstProxy;

    OSL_FORCEINLINE const Block<DataT, WidthT>& data() const
    {
        return m_ref_wide_data;
    }

    OSL_FORCEINLINE ConstProxy const operator[](int lane) const
    {
        return ConstProxy(m_ref_wide_data, lane);
    }

private:
    const Block<DataT, WidthT>& m_ref_wide_data;
};

template<typename ElementT, int ArrayLenT, int WidthT>
struct WideImpl<const ElementT[ArrayLenT], WidthT, true /*IsConstT */> {
    static constexpr int width    = WidthT;
    static constexpr int ArrayLen = ArrayLenT;
    static_assert(ArrayLen > 0, "OSL logic bug");
    typedef const ElementT ElementType;
    static_assert(std::is_const<ElementType>::value,
                  "unexpected compiler behavior");
    typedef ElementT NonConstElementType;
    typedef ElementType ArrayType[ArrayLen];

    explicit OSL_FORCEINLINE WideImpl(const void* ptr_wide_data)
        : m_array_of_wide_data(block_cast<ElementT, WidthT>(ptr_wide_data))
    {
    }

    explicit OSL_FORCEINLINE
    WideImpl(const Block<ElementT, WidthT>* array_of_wide_data)
        : m_array_of_wide_data(assume_aligned(array_of_wide_data))
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    WideImpl(const WideImpl& other) noexcept
        : m_array_of_wide_data(other.m_array_of_wide_data)
    {
    }

    static constexpr OSL_FORCEINLINE int length() { return ArrayLen; }


    typedef ConstWideArrayLaneProxy<ElementType, ArrayLen, WidthT> Proxy;

    OSL_FORCEINLINE Proxy const operator[](int lane) const
    {
        return Proxy(m_array_of_wide_data, lane);
    }

    OSL_FORCEINLINE Wide<ElementType, WidthT> get_element(int array_index) const
    {
        OSL_DASSERT(array_index < ArrayLen);
        return Wide<ElementType, WidthT>(m_array_of_wide_data[array_index]);
    }

private:
    const Block<ElementT, WidthT>* m_array_of_wide_data;
};

template<typename ElementT, int WidthT>
struct WideImpl<const ElementT[], WidthT, true /*IsConstT */> {
    static constexpr int width = WidthT;
    typedef const ElementT ElementType;
    static_assert(std::is_const<ElementType>::value,
                  "unexpected compiler behavior");
    typedef ElementT NonConstElementType;

    explicit OSL_FORCEINLINE WideImpl(const void* ptr_wide_data,
                                      int array_length)
        : m_array_of_wide_data(block_cast<ElementT, WidthT>(ptr_wide_data))
        , m_array_length(array_length)
    {
    }

    explicit OSL_FORCEINLINE
    WideImpl(const Block<ElementT, WidthT>* array_of_wide_data,
             int array_length)
        : m_array_of_wide_data(assume_aligned(array_of_wide_data))
        , m_array_length(array_length)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    WideImpl(const WideImpl& other) noexcept
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
    {
    }

    OSL_FORCEINLINE int length() { return m_array_length; }


    typedef ConstWideUnboundedArrayLaneProxy<ElementType, WidthT> Proxy;

    OSL_FORCEINLINE Proxy const operator[](int lane) const
    {
        return Proxy(m_array_of_wide_data, m_array_length, lane);
    }

    OSL_FORCEINLINE Wide<ElementType, WidthT> get_element(int array_index) const
    {
        OSL_DASSERT(array_index < m_array_length);
        return Wide<ElementType, WidthT>(m_array_of_wide_data[array_index]);
    }

private:
    const Block<ElementT, WidthT>* m_array_of_wide_data;
    int m_array_length;
};

template<typename ElementT, int WidthT>
struct WideImpl<const Dual2<ElementT>[], WidthT, true /*IsConstT */> {
    static constexpr int width = WidthT;
    typedef const Dual2<ElementT> ElementType;
    typedef Dual2<ElementT> NonConstElementType;

    explicit OSL_FORCEINLINE WideImpl(const void* ptr_wide_data,
                                      int array_length)
        : m_array_of_wide_data(block_cast<ElementT, WidthT>(ptr_wide_data))
        , m_array_length(array_length)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    WideImpl(const WideImpl& other) noexcept
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
    {
    }

    OSL_FORCEINLINE int length() { return m_array_length; }

    typedef ConstWideDual2UnboundedArrayLaneProxy<ElementT, WidthT> Proxy;

    OSL_FORCEINLINE Proxy const operator[](int lane_index) const
    {
        return Proxy(m_array_of_wide_data, m_array_length, lane_index);
    }

    // get_element doesn't work here as val, dx, dy will be separated
    // in memory by m_array_length.  Perhaps could add get_val(), get_dx(), get_dy()
    // similar to MaskedData in order to enable get_element.
private:
    const Block<ElementT, WidthT>* m_array_of_wide_data;
    int m_array_length;
};

}  // namespace pvt


#if OSL_INTEL_COMPILER || OSL_GNUC_VERSION
// Workaround for error #3466: inheriting constructors must be inherited from a direct base class
#    define __OSL_INHERIT_BASE_CTORS(DERIVED, BASE) \
        using Base = typename DERIVED::BASE;        \
        using Base::BASE;
#else
#    define __OSL_INHERIT_BASE_CTORS(DERIVED, BASE) \
        using Base = typename DERIVED::BASE;        \
        using Base::Base;
#endif

// Wide wraps a reference to Block and provides a proxy to access to DataT
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
template<typename DataT, int WidthT>
struct Wide : pvt::WideImpl<DataT, WidthT, std::is_const<DataT>::value> {
    __OSL_INHERIT_BASE_CTORS(Wide, WideImpl)
    static constexpr int width = WidthT;
};


namespace pvt {

template<typename ConstDataT, int WidthT> struct UniformAsWideImpl {
    static_assert(std::is_const<ConstDataT>::value,
                  "Only meant for const ConstDataT");
    static_assert(
        std::is_array<ConstDataT>::value == false,
        "Only meant for non-array ConstDataT, arrays are meant to use specialized UniformAsWideImpl");
    static constexpr int width = WidthT;
    typedef typename std::remove_const<ConstDataT>::type NonConstValueType;

    explicit OSL_FORCEINLINE UniformAsWideImpl(const void* ptr_uniform_data,
                                               int derivIndex = 0)
        : m_ref_uniform_data(
            reinterpret_cast<ConstDataT*>(ptr_uniform_data)[derivIndex])
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    UniformAsWideImpl(const UniformAsWideImpl& other) noexcept
        : m_ref_uniform_data(other.m_ref_uniform_data)
    {
    }

    OSL_FORCEINLINE ConstDataT& data() const { return m_ref_uniform_data; }


    OSL_FORCEINLINE ConstDataT& operator[](int /*lane*/) const
    {
        return m_ref_uniform_data;
    }

private:
    ConstDataT& m_ref_uniform_data;
};

template<typename ElementT, int WidthT>
struct UniformAsWideImpl<const ElementT[], WidthT> {
    static constexpr int width = WidthT;
    typedef const ElementT ElementType;
    typedef ElementT NonConstElementType;


    explicit OSL_FORCEINLINE UniformAsWideImpl(const void* ptr_data,
                                               int array_length)
        : m_array_of_data(reinterpret_cast<ElementType*>(ptr_data))
        , m_array_length(array_length)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    UniformAsWideImpl(const UniformAsWideImpl& other) noexcept
        : m_array_of_data(other.m_array_of_data)
        , m_array_length(other.m_array_length)
    {
    }

    struct LaneProxy {
    private:
        const ElementT* m_array_of_data;
        int m_array_length;

    public:
        explicit OSL_FORCEINLINE LaneProxy(const ElementT* array_of_data,
                                           int array_length)
            : m_array_of_data(array_of_data), m_array_length(array_length)
        {
        }

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        LaneProxy(const LaneProxy& other)
            : m_array_of_data(other.m_array_of_data)
            , m_array_length(other.m_array_length)
        {
        }

        OSL_FORCEINLINE int length() const { return m_array_length; }

        OSL_FORCEINLINE ElementType const& operator[](int array_index) const
        {
            OSL_DASSERT(array_index < m_array_length);
            return m_array_of_data[array_index];
        }
    };


    OSL_FORCEINLINE LaneProxy const operator[](int /*lane*/) const
    {
        return LaneProxy(m_array_of_data, m_array_length);
    }

private:
    ElementType* m_array_of_data;
    int m_array_length;
};

template<typename ElementT, int WidthT>
struct UniformAsWideImpl<const Dual2<ElementT>[], WidthT> {
    static constexpr int width = WidthT;
    typedef const Dual2<ElementT> ElementType;
    typedef Dual2<ElementT> NonConstElementType;

    explicit OSL_FORCEINLINE UniformAsWideImpl(const void* ptr_data,
                                               int array_length)
        : m_array_of_data(reinterpret_cast<const ElementT*>(ptr_data))
        , m_array_length(array_length)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    UniformAsWideImpl(const UniformAsWideImpl& other) noexcept
        : m_array_of_data(other.m_array_of_data)
        , m_array_length(other.m_array_length)
    {
    }


    struct LaneProxy {
    private:
        const ElementT* m_array_of_data;
        int m_array_length;

    public:
        explicit OSL_FORCEINLINE LaneProxy(const ElementT* array_of_data,
                                           int array_length)
            : m_array_of_data(array_of_data), m_array_length(array_length)
        {
        }

        // Must provide user defined copy constructor to
        // get compiler to be able to follow individual
        // data members through back to original object
        // when fully inlined the proxy should disappear
        OSL_FORCEINLINE
        LaneProxy(const LaneProxy& other)
            : m_array_of_data(other.m_array_of_data)
            , m_array_length(other.m_array_length)
        {
        }

        OSL_FORCEINLINE int length() const { return m_array_length; }

        OSL_FORCEINLINE const Dual2<ElementT> operator[](int array_index) const
        {
            OSL_DASSERT(array_index < m_array_length);
            ElementT lx  = m_array_of_data[array_index];
            ElementT ldx = (m_array_of_data + m_array_length)[array_index];
            ElementT ldy = (m_array_of_data
                            + (2 * m_array_length))[array_index];
            return Dual2<ElementT>(lx, ldx, ldy);
        }
    };

    OSL_FORCEINLINE LaneProxy const operator[](int /*lane*/) const
    {
        return LaneProxy(m_array_of_data, m_array_length);
    }

private:
    const ElementT* m_array_of_data;
    int m_array_length;
};
}  // namespace pvt

template<typename DataT, int WidthT>
struct UniformAsWide : public pvt::UniformAsWideImpl<DataT, WidthT> {
    static_assert(
        std::is_const<DataT>::value,
        "UniformAsWide<typename DataT> is only valid when DataT is const");
    static_assert(
        std::extent<DataT>::value == 0,
        "Only unbounded arrays[] are implemented, additional specializations can be added for fixed size arrays[#] if needed");
    __OSL_INHERIT_BASE_CTORS(UniformAsWide, UniformAsWideImpl)
};


// End users can add specialize wide for their own types
// and specialize traits to enable them to be used in the proxies
// NOTE: array detection is handled separately
template<typename DataT>
struct WideTraits;  // undefined, all types used should be specialized
//{
//static bool mathes(const TypeDesc &) { return false; }
//};

template<> struct WideTraits<float> {
    static bool matches(const TypeDesc& type_desc)
    {
        // NOTE: using bitwise & to avoid branches
        return (type_desc.basetype == TypeDesc::FLOAT)
               & (type_desc.aggregate == TypeDesc::SCALAR);
    }
};

template<> struct WideTraits<int> {
    static bool matches(const TypeDesc& type_desc)
    {
        // NOTE: using bitwise & to avoid branches
        return (type_desc.basetype == TypeDesc::INT)
               & (type_desc.aggregate == TypeDesc::SCALAR);
    }
};

template<> struct WideTraits<char*> {
    static bool matches(const TypeDesc& type_desc)
    {
        // NOTE: using bitwise & to avoid branches
        return (type_desc.basetype == TypeDesc::STRING)
               & (type_desc.aggregate == TypeDesc::SCALAR);
    }
};

template<> struct WideTraits<ustring> {
    static bool matches(const TypeDesc& type_desc)
    {
        // NOTE: using bitwise & to avoid branches
        return (type_desc.basetype == TypeDesc::STRING)
               & (type_desc.aggregate == TypeDesc::SCALAR);
    }
};

// We let Vec3 match any vector semantics as we don't have a seperate Point or Normal classes
template<> struct WideTraits<Vec3> {
    static bool matches(const TypeDesc& type_desc)
    {
        // NOTE: using bitwise & to avoid branches
        return (type_desc.basetype == TypeDesc::FLOAT)
               & (type_desc.aggregate == TypeDesc::VEC3);
    }
};

template<> struct WideTraits<Vec2> {
    static bool matches(const TypeDesc& type_desc)
    {
        // NOTE: using bitwise & to avoid branches
        return (type_desc.basetype == TypeDesc::FLOAT)
               & (type_desc.aggregate == TypeDesc::VEC2);
    }
};

template<> struct WideTraits<Color3> {
    static bool matches(const TypeDesc& type_desc)
    {
        // NOTE: using bitwise & to avoid branches
        return (type_desc.basetype == TypeDesc::FLOAT)
               & (type_desc.aggregate == TypeDesc::VEC3)
               & (type_desc.vecsemantics == TypeDesc::COLOR);
    }
};

template<> struct WideTraits<Matrix33> {
    static bool matches(const TypeDesc& type_desc)
    {
        // NOTE: using bitwise & to avoid branches
        return (type_desc.basetype == TypeDesc::FLOAT)
               & (type_desc.aggregate == TypeDesc::MATRIX33);
    }
};

template<> struct WideTraits<Matrix44> {
    static bool matches(const TypeDesc& type_desc)
    {
        // NOTE: using bitwise & to avoid branches
        return (type_desc.basetype == TypeDesc::FLOAT)
               & (type_desc.aggregate == TypeDesc::MATRIX44);
    }
};



namespace pvt {

template<typename DataT, int WidthT> struct MaskedLaneProxy {
    typedef DataT const ValueType;

    explicit OSL_FORCEINLINE
    MaskedLaneProxy(Block<DataT, WidthT>& ref_wide_data,
                    const Mask<WidthT>& mask, const int lane)
        : m_ref_wide_data(ref_wide_data), m_mask(mask), m_lane(lane)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    MaskedLaneProxy(const MaskedLaneProxy& other)
        : m_ref_wide_data(other.m_ref_wide_data)
        , m_mask(other.m_mask)
        , m_lane(other.m_lane)
    {
    }

    OSL_FORCEINLINE
    operator ValueType() const { return m_ref_wide_data.get(m_lane); }

    // TODO: Investigate if test outside of assignment is ok
    // or is it better to forward the mask down to the data
    // block to handle the mask per component
#ifdef __OSL_WIDE_MASK_AT_OBJECT_LEVEL
    OSL_FORCEINLINE const DataT& operator=(const DataT& value) const
    {
        if (m_mask[m_lane]) {
            m_ref_wide_data.set(m_lane, value);
        }
        return value;
    }
#else
    OSL_FORCEINLINE const DataT& operator=(const DataT& value) const
    {
        m_ref_wide_data.set(m_lane, value, m_mask[m_lane]);
        return value;
    }
#endif

    // Although having free helper functions
    // might be cleaner, we choose to expose
    // this functionality here to increase
    // visibility to end user whose IDE
    // might display these methods vs. free
    // functions
    OSL_FORCEINLINE bool is_on() const { return m_mask.is_on(m_lane); }

    OSL_FORCEINLINE bool is_off() const { return m_mask.is_off(m_lane); }

private:
    Block<DataT, WidthT>& m_ref_wide_data;
    const Mask<WidthT>& m_mask;
    const int m_lane;
};


template<typename DataT, int ArrayLenT, int WidthT>
struct MaskedArrayLaneProxy {
    explicit OSL_FORCEINLINE
    MaskedArrayLaneProxy(Block<DataT, WidthT>* array_of_wide_data,
                         const Mask<WidthT>& mask, const int lane)
        : m_array_of_wide_data(array_of_wide_data), m_mask(mask), m_lane(lane)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    MaskedArrayLaneProxy(const MaskedArrayLaneProxy& other)
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_mask(other.m_mask)
        , m_lane(other.m_lane)
    {
    }

    static constexpr OSL_FORCEINLINE int length() { return ArrayLenT; }

    // TODO: Investigate if test outside of assignment is ok
    // or is it better to forward the mask down to the data
    // block to handle the mask per component
#ifdef __OSL_WIDE_MASK_AT_OBJECT_LEVEL
    OSL_FORCEINLINE
    const MaskedArrayLaneProxy& operator=(const DataT (&value)[ArrayLenT]) const
    {
        if (m_mask[m_lane]) {
            static_foreach<ConstIndex, ArrayLenT>(
                [&](int i) { m_array_of_wide_data[i].set(m_lane, value[i]); });
        }
        return *this;
    }
#else
    OSL_FORCEINLINE
    const MaskedArrayLaneProxy& operator=(const DataT (&value)[ArrayLenT]) const
    {
        static_foreach<ConstIndex, ArrayLenT>([&](int i) {
            m_array_of_wide_data[i].set(m_lane, value[i], m_mask[m_lane]);
        });
        return *this;
    }
#endif

    // Although having free helper functions
    // might be cleaner, we choose to expose
    // this functionality here to increase
    // visibility to end user whose IDE
    // might display these methods vs. free
    // functions
    OSL_FORCEINLINE bool is_on() const { return m_mask.is_on(m_lane); }

    OSL_FORCEINLINE bool is_off() const { return m_mask.is_off(m_lane); }

    OSL_FORCEINLINE MaskedLaneProxy<DataT, WidthT>
    operator[](int array_index) const
    {
        return MaskedLaneProxy<DataT, WidthT>(m_array_of_wide_data[array_index],
                                              m_mask, m_lane);
    }

private:
    mutable Block<DataT, WidthT>* m_array_of_wide_data;
    const Mask<WidthT>& m_mask;
    const int m_lane;
};


template<typename DataT, int WidthT> struct MaskedUnboundedArrayLaneProxy {
    explicit OSL_FORCEINLINE
    MaskedUnboundedArrayLaneProxy(Block<DataT, WidthT>* array_of_wide_data,
                                  int array_length, const Mask<WidthT>& mask,
                                  int lane)
        : m_array_of_wide_data(array_of_wide_data)
        , m_array_length(array_length)
        , m_mask(mask)
        , m_lane(lane)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    MaskedUnboundedArrayLaneProxy(const MaskedUnboundedArrayLaneProxy& other)
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
        , m_mask(other.m_mask)
        , m_lane(other.m_lane)
    {
    }

    OSL_FORCEINLINE int length() const { return m_array_length; }

    // Although having free helper functions
    // might be cleaner, we choose to expose
    // this functionality here to increase
    // visibility to end user whose IDE
    // might display these methods vs. free
    // functions
    OSL_FORCEINLINE bool is_on() const { return m_mask.is_on(m_lane); }

    OSL_FORCEINLINE bool is_off() const { return m_mask.is_off(m_lane); }

    OSL_FORCEINLINE MaskedLaneProxy<DataT, WidthT>
    operator[](int array_index) const
    {
        OSL_DASSERT(array_index < m_array_length);
        return MaskedLaneProxy<DataT, WidthT>(m_array_of_wide_data[array_index],
                                              m_mask, m_lane);
    }


private:
    mutable Block<DataT, WidthT>* m_array_of_wide_data;
    int m_array_length;
    const Mask<WidthT>& m_mask;
    const int m_lane;
};

template<typename DataT, int WidthT> struct MaskedImpl {
    static constexpr int width = WidthT;
    typedef DataT ValueType;

    explicit OSL_FORCEINLINE MaskedImpl(void* ptr_wide_data, Mask<WidthT> mask,
                                        int derivIndex = 0)
        : m_ref_wide_data(*block_cast<DataT, WidthT>(ptr_wide_data, derivIndex))
        , m_mask(mask)
    {
    }

    explicit OSL_FORCEINLINE MaskedImpl(Block<DataT, WidthT>& ref_wide_data,
                                        Mask<WidthT> mask)
        : m_ref_wide_data(align_block_ref(ref_wide_data)), m_mask(mask)
    {
    }

    explicit OSL_FORCEINLINE MaskedImpl(const MaskedData<WidthT>& md,
                                        int derivIndex)
        : MaskedImpl(md.ptr(), md.mask(), derivIndex)
    {
    }

protected:
    explicit OSL_FORCEINLINE MaskedImpl(const MaskedImpl& other,
                                        Mask<WidthT> mask)
        : m_ref_wide_data(other.m_ref_wide_data), m_mask(mask)
    {
    }

    friend struct Masked<DataT, WidthT>;

    static OSL_FORCEINLINE bool supports(const MaskedData<WidthT>& md)
    {
        // NOTE: using bitwise & to avoid branches
        return (md.type().arraylen == 0)
               & WideTraits<DataT>::matches(md.type());
    }

public:
    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    MaskedImpl(const MaskedImpl& other) noexcept
        : m_ref_wide_data(other.m_ref_wide_data), m_mask(other.m_mask)
    {
    }



    OSL_FORCEINLINE Block<DataT, WidthT>& data() const
    {
        return m_ref_wide_data;
    }
    OSL_FORCEINLINE const Mask<WidthT>& mask() const { return m_mask; }

    typedef MaskedLaneProxy<DataT, WidthT> Proxy;

    OSL_FORCEINLINE Proxy operator[](int lane) const
    {
        return Proxy(m_ref_wide_data, m_mask, lane);
    }

    // Allow an ActiveLane to skip checking the mask
    typedef LaneProxy<DataT, WidthT> ActiveProxy;

    OSL_FORCEINLINE ActiveProxy operator[](ActiveLane lane) const
    {
        return ActiveProxy(m_ref_wide_data, lane.value());
    }

    // implicit conversion to constant read only access
    operator Wide<const DataT, WidthT>() const
    {
        return Wide<const DataT, WidthT>(m_ref_wide_data);
    }

private:
    Block<DataT, WidthT>& m_ref_wide_data;
    Mask<WidthT> m_mask;
};

template<typename ElementT, int ArrayLenT, int WidthT>
struct MaskedImpl<ElementT[ArrayLenT], WidthT> {
    static constexpr int width    = WidthT;
    static constexpr int ArrayLen = ArrayLenT;
    static_assert(ArrayLen > 0, "OSL logic bug");
    typedef ElementT ElementType;
    typedef ElementType ArrayType[ArrayLen];

    explicit OSL_FORCEINLINE MaskedImpl(void* ptr_wide_data, Mask<WidthT> mask,
                                        int derivIndex)
        : m_array_of_wide_data(&block_cast<ElementType, WidthT>(
            ptr_wide_data)[ArrayLen * derivIndex])
        , m_mask(mask)
    {
    }

    explicit OSL_FORCEINLINE MaskedImpl(const MaskedData<WidthT>& md,
                                        int derivIndex)
        : MaskedImpl(md.ptr(), md.mask(), derivIndex)
    {
    }

protected:
    explicit OSL_FORCEINLINE MaskedImpl(const MaskedImpl& other,
                                        Mask<WidthT> mask)
        : m_array_of_wide_data(other.m_array_of_wide_data), m_mask(mask)
    {
    }

    friend struct Masked<ElementT[ArrayLenT], WidthT>;

    static OSL_FORCEINLINE bool supports(const MaskedData<WidthT>& md)
    {
        // NOTE: using bitwise & to avoid branches
        return (md.type().arraylen == ArrayLen)
               & WideTraits<ElementType>::matches(md.type());
    }

public:
    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    MaskedImpl(const MaskedImpl& other) noexcept
        : m_array_of_wide_data(other.m_array_of_wide_data), m_mask(other.m_mask)
    {
    }


    static constexpr OSL_FORCEINLINE int length() { return ArrayLen; }

    typedef MaskedArrayLaneProxy<ElementType, ArrayLen, WidthT> Proxy;

    OSL_FORCEINLINE Proxy const operator[](int lane) const
    {
        return Proxy(m_array_of_wide_data, m_mask, lane);
    }

    OSL_FORCEINLINE Masked<ElementType, WidthT>
    get_element(int array_index) const
    {
        OSL_DASSERT(array_index < ArrayLen);
        return Masked<ElementType, WidthT>(m_array_of_wide_data[array_index],
                                           m_mask);
    }

private:
    mutable Block<ElementType, WidthT>* m_array_of_wide_data;
    Mask<WidthT> m_mask;
};

template<typename ElementT, int WidthT> struct MaskedImpl<ElementT[], WidthT> {
    static constexpr int width = WidthT;
    typedef ElementT ElementType;

    explicit OSL_FORCEINLINE MaskedImpl(void* ptr_wide_data, int array_length,
                                        Mask<WidthT> mask, int derivIndex)
        : m_array_of_wide_data(assume_aligned(&block_cast<ElementType, WidthT>(
            ptr_wide_data)[array_length * derivIndex]))
        , m_array_length(array_length)
        , m_mask(mask)
    {
    }

    explicit OSL_FORCEINLINE MaskedImpl(const MaskedData<WidthT>& md,
                                        int derivIndex)
        : MaskedImpl(md.ptr(), md.type().arraylen, md.mask(), derivIndex)
    {
    }

protected:
    OSL_FORCEINLINE
    MaskedImpl(const MaskedImpl& other, Mask<WidthT> mask)
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
        , m_mask(mask)
    {
    }

    friend struct Masked<ElementT[], WidthT>;

    static OSL_FORCEINLINE bool supports(const MaskedData<WidthT>& md)
    {
        // NOTE: using bitwise & to avoid branches
        return (md.type().arraylen != 0)
               & WideTraits<ElementType>::matches(md.type());
    }

public:
    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    MaskedImpl(const MaskedImpl& other) noexcept
        : m_array_of_wide_data(other.m_array_of_wide_data)
        , m_array_length(other.m_array_length)
        , m_mask(other.m_mask)
    {
    }

    OSL_FORCEINLINE int length() const { return m_array_length; }

    typedef MaskedUnboundedArrayLaneProxy<ElementType, WidthT> Proxy;

    OSL_FORCEINLINE Proxy const operator[](int lane) const
    {
        return Proxy(m_array_of_wide_data, m_array_length, m_mask, lane);
    }

    OSL_FORCEINLINE Masked<ElementType, WidthT>
    get_element(int array_index) const
    {
        OSL_DASSERT(array_index < m_array_length);
        return Masked<ElementType, WidthT>(m_array_of_wide_data[array_index],
                                           m_mask);
    }

    // implicit conversion to constant read only access
    operator Wide<const ElementT[], WidthT>() const
    {
        return Wide<const ElementT[], WidthT>(m_array_of_wide_data,
                                              m_array_length);
    }

private:
    mutable Block<ElementType, WidthT>* m_array_of_wide_data;
    int m_array_length;
    Mask<WidthT> m_mask;
};


}  // namespace pvt


// Masked has a reference to Block and Mask value to indicate which data
// lanes are active inside the Block.
// NOTE: Masks values are not stored with/inside Blocks as they
// are a result of shader control flow/logic often originate
// on the stack.
//
// Provides Proxy access to DataT for an individual data lane
// inside the Block.  The proxy will ignore assignments
// to inactive data lanes.  Users cannot forget to test
// the mask, because the Proxy does it for them.
// Handles DataT being bounded array [13], ie: Masked<float[13], 16>
// Handles DataT being unbounded array [], ie: Masked<float[], 16>
// DataT must NOT be const.
// Implementations should support the following interface:
//{
//    static constexpr int width = WidthT;
//
//    impl-defined-proxy operator[](int lane);
//
//    // When DataT is ElementType[] unbounded array
//    int length() const; // length of unbounded array
//
//    // When DataT is ElementType[] unbounded array
//    // provide Wide access to individual array element
//    Wide<ElementType, WidthT> get_element(int array_index) const
//
//    // Build an accessor combining current mask with another
//    Masked operator & (const Mask<WidthT> &) const
//
//    // Test MaskedData (could by any data type) if it match this DataT
//    static bool is(const MaskedData<WidthT> &)
//};
template<typename DataT, int WidthT>
struct Masked : public pvt::MaskedImpl<DataT, WidthT> {
    static_assert(std::is_const<DataT>::value == false,
                  "Masked<> is only valid when DataT is NOT const");

    __OSL_INHERIT_BASE_CTORS(Masked, MaskedImpl)

    // Allow implicit construction
    OSL_FORCEINLINE
    Masked(const MaskedData<WidthT>& md) noexcept
        : pvt::MaskedImpl<DataT, WidthT>(md, 0 /*derivIndex*/)
    {
    }

    OSL_FORCEINLINE Masked operator&(const Mask<WidthT>& conjunction_mask) const
    {
        return Masked(*this, pvt::MaskedImpl<DataT, WidthT>::mask()
                                 & conjunction_mask);
    }

    static OSL_FORCEINLINE bool is(const MaskedData<WidthT>& md)
    {
        return pvt::MaskedImpl<DataT, WidthT>::supports(md);
    }
};

namespace pvt {
template<typename DataT, int WidthT, int DerivIndexT>
struct MaskedDeriv : public Masked<DataT, WidthT> {
    // Allow implicit construction
    OSL_FORCEINLINE
    MaskedDeriv(const MaskedData<WidthT>& md)
        : Masked<DataT, WidthT>(md, DerivIndexT)
    {
    }

    explicit OSL_FORCEINLINE MaskedDeriv(void* ptr_wide_data, Mask<WidthT> mask)
        : Masked<DataT, WidthT>(ptr_wide_data, mask, DerivIndexT)
    {
    }

    OSL_FORCEINLINE MaskedDeriv operator&(const Mask<WidthT>& mask) const
    {
        return MaskedDeriv(*this, Masked<DataT, WidthT>::mask() & mask);
    }
};
}  // namespace pvt

// Block<Dual2<DataT>> actually stores val, dx, dy in separate adjacent Blocks.
// Masked<> should not be instantiated with a Dual2, but instead
// use these additional wrappers to get at derivative data
//
//     template <typename DataT, int WidthT>
//     struct MaskedDx;
//
//     template <typename DataT, int WidthT>
//     struct MaskedDy;
//
// Same interface as Masked, but treats Block & as array and accesses
// Block[1] for Dx, Block[2] for Dy
template<typename DataT, int WidthT>
using MaskedDx = pvt::MaskedDeriv<DataT, WidthT, 1 /*DerivIndexT*/>;
template<typename DataT, int WidthT>
using MaskedDy = pvt::MaskedDeriv<DataT, WidthT, 2 /*DerivIndexT*/>;


template<typename DataT, int WidthT>
OSL_FORCEINLINE void
assign_all(Masked<DataT, WidthT> wdest, const DataT& value)
{
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
        for (int i = 0; i < WidthT; ++i) {
            wdest[i] = value;
        }
    }
}

template<typename DataT, int WidthT>
OSL_FORCEINLINE void
assign_all(Masked<DataT[], WidthT> wide_data, const DataT* value_array)
{
    for (int array_index = 0; array_index < wide_data.length(); ++array_index) {
        assign_all(wide_data.get_element(array_index),
                   value_array[array_index]);
    }
}


template<typename DataT, int WidthT, typename FunctorT>
OSL_FORCEINLINE void
foreach_unique(Wide<DataT, WidthT> wdata, Mask<WidthT> data_mask, FunctorT f)
{
    OSL_DASSERT(data_mask.any_on());
    // The following control flow assumes at least 1 data lane is active in the data_mask
    Mask<WidthT> remaining_mask(data_mask);
    do {
        ActiveLane lead_lane(remaining_mask.first_on());
        DataT lead_data = wdata[lead_lane];
        Mask<WidthT> matching_lanes(false);
        OSL_FORCEINLINE_BLOCK
        {
            OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
            for (int lane = 0; lane < WidthT; ++lane) {
                // NOTE: the comparison ignores the remaining_mask
                bool lane_matches = (lead_data == wdata[lane]);
                // NOTE: using bitwise & to avoid branches
                if (lane_matches & remaining_mask[lane]) {
                    matching_lanes.set_on(lane);
                }
            }
        }

        f(lead_data, matching_lanes);
        remaining_mask &= ~matching_lanes;
    } while (remaining_mask.any_on());
}


// MaskedData is a combination of a pointer to a Block unknown DataT,
// a TypeDesc to identify it, and a flag to indicate if derivatives are present.
// Used to pass a Block of data whose type could anything to a function.
// The receiving function must test a specific set Masked<DataT> to see if
// it can be constructed from the MaskedData.
// Although the underlying TypeDesc has its own ways of being tested,
// we have provided trait classes to perform the testing based on DataT.
// IE:  void myFunction(MaskedData<16> any) {
//          if (Masked<Vec3>::is(any) {
//              Masked<Vec3> vecVal(any);
//              process(vecVal);
//              if (any.has_derivs()) {
//                  MaskedDx<Vec3> vecDx(any);
//                  MaskedDy<Vec3> vecDy(any);
//                  processDerivs(vecDx, vecDy);
//              }
//          } else if (Masked<int[2]>::is(any) {
//              Masked<int[2]> resolution(any);
//              process(resolution);
//          }
//      }
template<int WidthT> class MaskedData {
    mutable void* m_ptr;
    TypeDesc m_type;
    Mask<WidthT> m_mask;
    bool m_has_derivs;

public:
    static constexpr int width = WidthT;

    MaskedData() = delete;

    explicit OSL_FORCEINLINE MaskedData(TypeDesc type, bool has_derivs,
                                        Mask<WidthT> mask, void* ptr)
        : m_ptr(assume_aligned<VecReg<WidthT>::alignment>(ptr))
        , m_type(type)
        , m_mask(mask)
        , m_has_derivs(has_derivs)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE MaskedData(const MaskedData& other)
        : m_ptr(other.m_ptr)
        , m_type(other.m_type)
        , m_mask(other.m_mask)
        , m_has_derivs(other.m_has_derivs)
    {
    }

    OSL_FORCEINLINE void* ptr() const { return m_ptr; }
    OSL_FORCEINLINE const TypeDesc& type() const { return m_type; }
    OSL_FORCEINLINE bool has_derivs() const { return m_has_derivs; }
    OSL_FORCEINLINE const Mask<WidthT>& mask() const { return m_mask; }
    OSL_FORCEINLINE bool valid() const { return m_ptr != nullptr; }


    OSL_FORCEINLINE MaskedData operator&(const Mask<WidthT>& mask) const
    {
        return MaskedData(m_type, m_has_derivs, m_mask & mask, m_ptr);
    }

    OSL_NOINLINE size_t val_size_in_bytes() const;
};


// For consistency, for passing unknown uniform data, RefData can be used
// in a similar fashion to MaskedData
// RefData is a combination of a pointer to uniform unknown DataT,
// a TypeDesc to identify it, and a flag to indicate if derivatives are present.
// Used to pass data whose type could anything to a function.
// The receiving function must test a specific set Ref<DataT> to see if
// it can be constructed from the RefData.
// Although the underlying TypeDesc has its own ways of being tested,
// we have provided trait classes to perform the testing based on DataT.
// IE:  void myFunction(RefData any) {
//          if (Ref<Vec3>::is(any) {
//              Ref<Vec3> vecVal(any);
//              process(vecVal);
//              if (any.has_derivs()) {
//                  RefDx<Vec3> vecDx(any);
//                  RefDy<Vec3> vecDy(any);
//                  processDerivs(vecDx, vecDy);
//              }
//          } else if (Ref<ustring>::is(any) {
//              Ref<ustring> msg(any);
//              process(msg);
//          }
//      }
class RefData {
    mutable void* m_ptr;
    TypeDesc m_type;
    bool m_has_derivs;

public:
    RefData() = delete;

    explicit OSL_FORCEINLINE RefData(TypeDesc type, bool has_derivs, void* ptr)
        : m_ptr(ptr), m_type(type), m_has_derivs(has_derivs)
    {
    }

    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE RefData(const RefData& other)
        : m_ptr(other.m_ptr)
        , m_type(other.m_type)
        , m_has_derivs(other.m_has_derivs)
    {
    }

    OSL_FORCEINLINE void* ptr() const { return m_ptr; }
    OSL_FORCEINLINE TypeDesc type() const { return m_type; }
    OSL_FORCEINLINE bool has_derivs() const { return m_has_derivs; }
    OSL_FORCEINLINE bool valid() const { return m_ptr != nullptr; }

    // To access underlying data in type safe manner use
    // bool Ref<DataT>::is(const RefData &)
    // Ref<DataT>(const RefData &)
    // RefDx<DataT>(const RefData &)
    // RefDy<DataT>(const RefData &)
};

namespace pvt {

// Pretty much just allows "auto" to be used on the stack to
// keep a reference vs. a copy of DataT
template<typename DataT> struct RefImpl {
    static_assert(std::is_const<DataT>::value == false,
                  "Logic Bug:  Only meant for non-const DataT");
    static_assert(
        std::is_array<DataT>::value == false,
        "Logic Bug:  Only meant for non-array DataT, arrays are meant to use specialized RefImpl");

    explicit OSL_FORCEINLINE RefImpl(DataT& ref_data) : m_ref_data(ref_data) {}

    explicit OSL_FORCEINLINE RefImpl(const RefData& rd, int derivIndex)
        : RefImpl(reinterpret_cast<DataT*>(rd.ptr())[derivIndex])
    {
    }

protected:
    friend struct Ref<DataT>;

    static OSL_FORCEINLINE bool supports(const RefData& rd)
    {
        // NOTE: using bitwise & to avoid branches
        return (rd.type().arraylen == 0)
               & WideTraits<DataT>::matches(rd.type());
    }

public:
    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    RefImpl(const RefImpl& other) : m_ref_data(other.m_ref_data) {}


    OSL_FORCEINLINE
    operator DataT&() const { return m_ref_data; }

    OSL_FORCEINLINE const DataT& operator=(const DataT& value) const
    {
        m_ref_data = value;
        return value;
    }

private:
    DataT& m_ref_data;
};


template<typename ElementT, int ArrayLenT> struct RefImpl<ElementT[ArrayLenT]> {
    static constexpr int ArrayLen = ArrayLenT;
    static_assert(ArrayLen > 0, "OSL logic bug");
    typedef ElementT ElementType;
    typedef ElementType ArrayType[ArrayLen];

    explicit OSL_FORCEINLINE RefImpl(ArrayType& ref_array_data)
        : m_ref_array_data(ref_array_data)
    {
    }

    explicit OSL_FORCEINLINE RefImpl(const RefData& rd, int derivIndex)
        : RefImpl(reinterpret_cast<ArrayType*>(rd.ptr())[derivIndex])
    {
    }

protected:
    friend struct Ref<ArrayType>;

    static OSL_FORCEINLINE bool supports(const RefData& rd)
    {
        // NOTE: using bitwise & to avoid branches
        return (rd.type().arraylen == ArrayLen)
               & WideTraits<ElementType>::matches(rd.type());
    }

public:
    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    RefImpl(const RefImpl& other) : m_ref_array_data(other.m_ref_array_data) {}

    OSL_FORCEINLINE constexpr int length() const { return ArrayLen; }

    OSL_FORCEINLINE
    const RefImpl& operator=(const ArrayType& value) const
    {
        static_foreach<ConstIndex, ArrayLenT>(
            [&](int i) { m_ref_array_data[i] = value[i]; });
        return *this;
    }

    OSL_FORCEINLINE
    operator ArrayType&() const { return m_ref_array_data; }


    OSL_FORCEINLINE ElementType& operator[](int array_index) const
    {
        OSL_DASSERT(array_index >= 0 && array_index < ArrayLen);
        return m_ref_array_data[array_index];
    }

private:
    ArrayType& m_ref_array_data;
};



template<typename ElementT> struct RefImpl<ElementT[]> {
    typedef ElementT ElementType;

    explicit OSL_FORCEINLINE RefImpl(ElementType* array_data, int array_length)
        : m_array_data(array_data), m_array_length(array_length)
    {
    }

    explicit OSL_FORCEINLINE RefImpl(const RefData& rd, int derivIndex)
        : RefImpl(&(reinterpret_cast<ElementType*>(
                      rd.ptr())[derivIndex * rd.type().arraylen]),
                  rd.type().arraylen)
    {
    }

protected:
    friend struct Ref<ElementT[]>;

    static OSL_FORCEINLINE bool supports(const RefData& rd)
    {
        // NOTE: using bitwise & to avoid branches
        return (rd.type().arraylen != 0)
               & WideTraits<ElementType>::matches(rd.type());
    }

public:
    // Must provide user defined copy constructor to
    // get compiler to be able to follow individual
    // data members through back to original object
    // when fully inlined the proxy should disappear
    OSL_FORCEINLINE
    RefImpl(const RefImpl& other)
        : m_array_data(other.m_array_data), m_array_length(other.m_array_length)
    {
    }

    OSL_FORCEINLINE int length() const { return m_array_length; }

    // Omit conversion operator & assignment for unbounded array

    OSL_FORCEINLINE ElementType& operator[](int array_index) const
    {
        OSL_DASSERT(array_index >= 0 && array_index < m_array_length);
        return m_array_data[array_index];
    }

private:
    mutable ElementType* m_array_data;
    int m_array_length;
};

}  // namespace pvt


// Reference to DataT
//
// Provides type specific Proxy access to DataT.
// Usually constructed from RefData.
// Handles DataT being bounded array [13], ie: Ref<float[13]>
// Handles DataT being unbounded array [], ie: Ref<float[]>
// DataT must NOT be const.
// Implementations should support the following interface:
//{
//    operator DataT & () const
//    impl-defined operator = (const DataT & value) const
//
//    // When DataT is ElementType[] or ElementType[int]
//    int length() const; // length of unbounded array
//
//    // When DataT is ElementType[] or ElementType[int]
//    // provide Wide access to individual array element
//    ElementType & operator[](int array_index) const
//
//    // Test RefData (could by any data type) if it match this DataT
//    static bool is(const RefData &)
//};
template<typename DataT> struct Ref : public pvt::RefImpl<DataT> {
    __OSL_INHERIT_BASE_CTORS(Ref, RefImpl)
    using Base::operator=;

    // Allow implicit construction
    OSL_FORCEINLINE
    Ref(const RefData& rd) : pvt::RefImpl<DataT>(rd, 0 /*derivIndex*/) {}

    static OSL_FORCEINLINE bool is(const RefData& rd)
    {
        return pvt::RefImpl<DataT>::supports(rd);
    }
};

namespace pvt {
template<typename DataT, int DerivIndexT> struct RefDeriv : public Ref<DataT> {
    // Allow implicit construction
    OSL_FORCEINLINE
    RefDeriv(const RefData& rd) : Ref<DataT>(rd, DerivIndexT /*derivIndex*/) {}

    using Base          = typename RefDeriv::Ref;
    using Base::operator=;
};
}  // namespace pvt


// Ref<> should not be instantiated with a Dual2, but instead
// use these additional wrappers to get at derivative data
//
//     template <typename DataT>
//     struct RefDx;
//
//     template <typename DataT>
//     struct RefDy;
//
// Same interface as Ref, but treats DataT & as array and accesses
// DataT*[1] for Dx, DataT*[2] for Dy
template<typename DataT> using RefDx = pvt::RefDeriv<DataT, 1 /*DerivIndexT*/>;
template<typename DataT> using RefDy = pvt::RefDeriv<DataT, 2 /*DerivIndexT*/>;


template<typename LaneProxyT>
typename LaneProxyT::ValueType const
unproxy(const LaneProxyT& proxy)
{
    return proxy.operator typename LaneProxyT::ValueType();
}

template<typename DataT, int WidthT>
OSL_FORCEINLINE bool
testIfAnyLaneIsNonZero(const Wide<DataT, WidthT>& wvalues)
{
#if OSL_NON_INTEL_CLANG
    int anyLaneIsOn = 0;
    OSL_OMP_PRAGMA(omp simd simdlen(WidthT) reduction(max : anyLaneIsOn))
    for (int i = 0; i < WidthT; ++i) {
        if (wvalues[i] > anyLaneIsOn)
            anyLaneIsOn = wvalues[i];
    }
    return anyLaneIsOn;
#else
    // NOTE: do not explicitly vectorize as it would require a
    // reduction.  Instead let compiler optimize this itself.
    bool anyLaneIsOn = false;
    for (int i = 0; i < WidthT; ++i) {
        if (wvalues[i] != DataT(0))
            anyLaneIsOn = true;
    }
    return anyLaneIsOn;
#endif
}

// The rest of MaskedData implementation that depends on
// Masked<DataT> being defined
template<int WidthT>
OSL_NOINLINE size_t
MaskedData<WidthT>::val_size_in_bytes() const
{
    if (Masked<ustring, WidthT>::is(*this)) {
        return sizeof(Block<ustring, WidthT>);
    }
    if (Masked<int, WidthT>::is(*this)) {
        return sizeof(Block<int, WidthT>);
    }
    if (Masked<float, WidthT>::is(*this)) {
        return sizeof(Block<float, WidthT>);
    }
    if (Masked<Vec3, WidthT>::is(*this)) {
        return sizeof(Block<Vec3, WidthT>);
    }
    if (Masked<Matrix44, WidthT>::is(*this)) {
        return sizeof(Block<Matrix44, WidthT>);
    }


    if (Masked<ustring[], WidthT>::is(*this)) {
        return sizeof(Block<ustring, WidthT>) * m_type.arraylen;
    }
    if (Masked<int[], WidthT>::is(*this)) {
        return sizeof(Block<int, WidthT>) * m_type.arraylen;
    }
    if (Masked<float[], WidthT>::is(*this)) {
        return sizeof(Block<float, WidthT>) * m_type.arraylen;
    }
    if (Masked<Vec3[], WidthT>::is(*this)) {
        return sizeof(Block<Vec3, WidthT>) * m_type.arraylen;
    }
    if (Masked<Matrix44[], WidthT>::is(*this)) {
        return sizeof(Block<Matrix44, WidthT>) * m_type.arraylen;
    }


    // Not exposed in OSL language itself
    if (Masked<Vec2, WidthT>::is(*this)) {
        return sizeof(Block<Vec2, WidthT>);
    }
    OSL_DASSERT(0 && "unsupported or incomplete for TypeDesc");
}


#define __OSL_USING_WIDE(WIDTH_OF_OSL_DATA)                             \
    template<typename DataT>                                            \
    using Block = OSL_NAMESPACE::Block<DataT, WIDTH_OF_OSL_DATA>;       \
                                                                        \
    using Mask = OSL_NAMESPACE::Mask<WIDTH_OF_OSL_DATA>;                \
                                                                        \
    using MaskedData = OSL_NAMESPACE::MaskedData<WIDTH_OF_OSL_DATA>;    \
                                                                        \
    template<typename DataT>                                            \
    using Wide = OSL_NAMESPACE::Wide<DataT, WIDTH_OF_OSL_DATA>;         \
                                                                        \
    template<typename DataT>                                            \
    using UniformAsWide                                                 \
        = OSL_NAMESPACE::UniformAsWide<DataT, WIDTH_OF_OSL_DATA>;       \
                                                                        \
    template<typename DataT>                                            \
    using Masked = OSL_NAMESPACE::Masked<DataT, WIDTH_OF_OSL_DATA>;     \
    template<typename DataT>                                            \
    using MaskedDx = OSL_NAMESPACE::MaskedDx<DataT, WIDTH_OF_OSL_DATA>; \
    template<typename DataT>                                            \
    using MaskedDy = OSL_NAMESPACE::MaskedDy<DataT, WIDTH_OF_OSL_DATA>;


// Use inside of
//     namespace __OSL_WIDE_PVT {
//         OSL_USING_DATA_WIDTH(__OSL_WIDTH)
//
// or inside a class definition
//     template<int WidthT> class MyRendererServices {
//         OSL_USING_DATA_WIDTH(WidthT)
//
// to create template alias and typedefs for all of
// these wrappers with WidthT parameter hardcoded
#define OSL_USING_DATA_WIDTH(WIDTH_OF_OSL_DATA) \
    __OSL_USING_WIDE(WIDTH_OF_OSL_DATA)

#undef __OSL_INHERIT_BASE_CTORS

OSL_NAMESPACE_EXIT
