// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <immintrin.h>
#include <type_traits>

#include <OSL/oslconfig.h>

OSL_NAMESPACE_ENTER


// clang-format off

// Define popcount and countr_zero
#if OSL_CPLUSPLUS_VERSION >= 20

// For C++20 and beyond, these are in the standard library
using std::popcount;
using std::countr_zero;

#elif OSL_INTEL_COMPILER

OSL_FORCEINLINE int popcount(uint32_t x) noexcept { return _mm_popcnt_u32(x);}
OSL_FORCEINLINE int popcount(uint64_t x) noexcept { return _mm_popcnt_u64(x); }
OSL_FORCEINLINE int countr_zero(uint32_t x) noexcept { return _bit_scan_forward(x); }
OSL_FORCEINLINE int countr_zero(uint64_t x) noexcept {
    unsigned __int32 index;
    _BitScanForward64(&index, x);
    return static_cast<int>(index);
}

#elif defined(__GNUC__) || defined(__clang__)

OSL_FORCEINLINE int popcount(uint32_t x) noexcept { return __builtin_popcount(x); }
OSL_FORCEINLINE int popcount(uint64_t x) noexcept { return __builtin_popcountll(x); }
OSL_FORCEINLINE int countr_zero(uint32_t x) noexcept { return __builtin_ctz(x); }
OSL_FORCEINLINE int countr_zero(uint64_t x) noexcept { return __builtin_ctzll(x); }

#elif defined(_MSC_VER)

OSL_FORCEINLINE int popcount(uint32_t x) noexcept { return static_cast<int>(__popcnt(x)); }
OSL_FORCEINLINE int popcount(uint64_t x) noexcept { return static_cast<int>(__popcnt64(x)); }
OSL_FORCEINLINE int countr_zero(uint32_t x) noexcept {
    unsigned long index;
    _BitScanForward(&index, x);
    return static_cast<int>(index);
}
OSL_FORCEINLINE int countr_zero(uint64_t x) noexcept {
    unsigned long index;
    _BitScanForward64(&index, x);
    return static_cast<int>(index);
}

#else
#    error "popcount and coutr_zero implementations needed for this compiler"
#endif

// clang-format on



// Simple wrapper to identify a single lane index vs. a mask_value
class Lane {
    const int m_index;

public:
    explicit OSL_FORCEINLINE Lane(int index) : m_index(index) {}

    Lane() = delete;

    OSL_FORCEINLINE Lane(const Lane& other) : m_index(other.m_index) {}

    OSL_FORCEINLINE int value() const { return m_index; }

    OSL_FORCEINLINE
    operator int() const { return m_index; }
};

// Simple wrapper to identify an active lane
// Active lanes will bypass mask testing during assignments to Masked::LaneProxy's
// But be careful if you ever have two Masked::LaneProxy's with
// different masks
class ActiveLane : public Lane {
public:
    explicit OSL_FORCEINLINE ActiveLane(int index) : Lane(index) {}

    ActiveLane() = delete;

    OSL_FORCEINLINE ActiveLane(const ActiveLane& other) : Lane(other) {}
};

template<int WidthT> class Mask {
    typedef unsigned short Value16Type;
    static_assert(sizeof(Value16Type) == 2, "unexpected platform");

    typedef unsigned int Value32Type;
    static_assert(sizeof(Value32Type) == 4, "unexpected platform");

    typedef unsigned long long Value64Type;
    static_assert(sizeof(Value64Type) == 8, "unexpected platform");

    typedef
        typename std::conditional<WidthT <= 32, Value32Type, Value64Type>::type
            Value32or64Type;

public:
#if 0  // Enable 16bit integer storage of masks, vs 32bit.
    typedef typename std::conditional<WidthT <= 16,
                                      Value16Type,
                                      Value32or64Type>::type ValueType;
#else
    typedef Value32or64Type ValueType;
#endif

    static constexpr int width = WidthT;

protected:
    static constexpr int value_width = sizeof(ValueType) * 8;
    static_assert(value_width >= WidthT, "unsupported WidthT");
    static constexpr ValueType valid_bits
        = static_cast<ValueType>(0xFFFFFFFFFFFFFFFF) >> (value_width - WidthT);

public:
    OSL_FORCEINLINE Mask() {}

    explicit OSL_FORCEINLINE Mask(Lane lane) : m_value(1 << lane.value()) {}

    explicit OSL_FORCEINLINE Mask(bool all_on_or_off)
        : m_value((all_on_or_off) ? valid_bits : 0)
    {
    }

    explicit constexpr OSL_FORCEINLINE Mask(std::false_type) : m_value(0) {}

    explicit constexpr OSL_FORCEINLINE Mask(std::true_type)
        : m_value(valid_bits)
    {
    }

    explicit OSL_FORCEINLINE Mask(Value16Type value_)
        : m_value(static_cast<ValueType>(value_))
    {
    }

    explicit OSL_FORCEINLINE Mask(Value32Type value_)
        : m_value(static_cast<ValueType>(value_))
    {
    }

    explicit OSL_FORCEINLINE Mask(Value64Type value_)
        : m_value(static_cast<ValueType>(value_))
    {
    }

    explicit OSL_FORCEINLINE Mask(int value_)
        : m_value(static_cast<ValueType>(value_))
    {
    }

    OSL_FORCEINLINE Mask(const Mask& other) : m_value(other.m_value) {}

    template<int OtherWidthT,
             typename = pvt::enable_if_type<(OtherWidthT < WidthT)>>
    explicit OSL_FORCEINLINE Mask(const Mask<OtherWidthT>& other)
        : m_value(static_cast<ValueType>(other.value()))
    {
    }


    OSL_FORCEINLINE ValueType value() const { return m_value; }

    // count number of active bits
    OSL_FORCEINLINE int count() const { return OSL::popcount(m_value); }

    // NOTE: undefined result if no bits are on
    OSL_FORCEINLINE int first_on() const { return OSL::countr_zero(m_value); }

    OSL_FORCEINLINE Mask invert() const
    {
        return Mask((~m_value) & valid_bits);
    }

    // Test only, don't allow assignment to force
    // more verbose set_on or set_off to be used
    // NOTE:  As the actual lane value is embedded
    // inside an integral type, we would have to
    // return a proxy which could complicate
    // codegen, so keeping it simple(r)
    OSL_FORCEINLINE bool operator[](int lane) const
    {
        // __assume(lane >= 0 && lane < width);

        //return (m_value & (1<<lane))==(1<<lane);
        //return (m_value >>lane) & 1;
        // From testing code generation this is the preferred form
        return (m_value & (1 << lane));
    }


    OSL_FORCEINLINE bool is_on(int lane) const
    {
        // From testing code generation this is the preferred form
        //return (m_value & (1<<lane))==(1<<lane);
        return (m_value & (1 << lane));
    }

    OSL_FORCEINLINE bool is_off(int lane) const
    {
        // From testing code generation this is the preferred form
        return (m_value & (1 << lane)) == 0;
    }

    OSL_FORCEINLINE bool all_on() const
    {
        // TODO:  is this more expensive than == ?
        return (m_value >= valid_bits);
    }

    OSL_FORCEINLINE bool all_off() const
    {
        return (m_value == static_cast<ValueType>(0));
    }

    OSL_FORCEINLINE bool any_on() const
    {
        return (m_value != static_cast<ValueType>(0));
    }

    OSL_FORCEINLINE bool any_off() const { return (m_value < valid_bits); }

    OSL_FORCEINLINE bool any_off(const Mask& mask) const
    {
        return m_value != (m_value & mask.m_value);
    }

    // Setters
    // For SIMD loops, set_on and set_off work better
    // than a generic set(lane,flag).
    // And in most all cases, the starting state
    // for the mask was all on or all off,
    // So really only set_on or set_off is required
    // Choose to not provide a generic set(int lane, bool flag)

    OSL_FORCEINLINE void set_on(int lane) { m_value |= (1 << lane); }

    OSL_FORCEINLINE void set_on_if(int lane, bool cond)
    {
        m_value |= (cond << lane);
    }

    OSL_FORCEINLINE void set_all_on() { m_value = valid_bits; }
    OSL_FORCEINLINE void set_count_on(int count)
    {
        m_value = valid_bits >> (width - count);
    }

    OSL_FORCEINLINE void set_off(int lane) { m_value &= (~(1 << lane)); }

    OSL_FORCEINLINE void set_off_if(int lane, bool cond)
    {
        m_value &= (~(cond << lane));
    }

    OSL_FORCEINLINE void set_all_off() { m_value = static_cast<ValueType>(0); }

    OSL_FORCEINLINE bool operator==(const Mask& other) const
    {
        return m_value == other.m_value;
    }

    OSL_FORCEINLINE bool operator!=(const Mask& other) const
    {
        return m_value != other.m_value;
    }

    OSL_FORCEINLINE Mask& operator&=(const Mask& other)
    {
        m_value = m_value & other.m_value;
        return *this;
    }

    OSL_FORCEINLINE Mask& operator|=(const Mask& other)
    {
        m_value = m_value | other.m_value;
        return *this;
    }

    OSL_FORCEINLINE Mask operator&(const Mask& other) const
    {
        return Mask(m_value & other.m_value);
    }

    OSL_FORCEINLINE Mask operator|(const Mask& other) const
    {
        return Mask(m_value | other.m_value);
    }

    OSL_FORCEINLINE Mask operator~() const { return invert(); }


    template<int MinOccupancyT, int MaxOccupancyT = width, typename FunctorT>
    OSL_FORCEINLINE void foreach (FunctorT f) const
    {
        // Expect compile time dead code elimination to skip this when possible
        if (MaxOccupancyT == 0)
            return;
        // Expect compile time dead code elimination to skip this when possible
        if (MinOccupancyT == 0) {
            if (all_off())
                return;
        }
        OSL_DASSERT(any_on());
        // Expect compile time dead code elimination to emit
        // one branch or the other
        if (MaxOccupancyT == 1) {
            ActiveLane active_lane(first_on());
            f(active_lane);
        } else {
            Mask m(m_value);
            do {
                ActiveLane active_lane(m.first_on());
                f(active_lane);
                m.set_off(active_lane);
            } while (m.any_on());
        }
    }

    // Serially apply functor f to each
    // lane active in the Mask
    template<typename FunctorT> OSL_FORCEINLINE void foreach (FunctorT f) const
    {
        foreach
            <0, width, FunctorT>(f);
    }

    // non-inlined version to isolate inherently serial codegen
    // of functor f from other call site whose code gen might
    // be SIMD in nature.  Not inlining prevents optimizer from
    // mixing serial code gen with call site which could inhibit
    // ability to generate SIMD code.
    template<int MinOccupancyT, int MaxOccupancyT = width, typename FunctorT>
    OSL_NOINLINE void invoke_foreach(FunctorT f) const;

    template<typename FunctorT>
    OSL_NOINLINE void invoke_foreach(FunctorT f) const;

    // Treat m_value as private, but access is needed for #pragma's to reference it in reduction clauses
    ValueType m_value;
};

template<int WidthT>
template<typename FunctorT>
void
Mask<WidthT>::invoke_foreach(FunctorT f) const
{
    foreach
        <0, width, FunctorT>(f);
}

template<int WidthT>
template<int MinOccupancyT, int MaxOccupancyT, typename FunctorT>
void
Mask<WidthT>::invoke_foreach(FunctorT f) const
{
    foreach
        <MinOccupancyT, MaxOccupancyT, FunctorT>(f);
}


OSL_NAMESPACE_EXIT
