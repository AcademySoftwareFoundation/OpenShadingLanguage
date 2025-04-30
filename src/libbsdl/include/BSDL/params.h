// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/config.h>

#include <Imath/ImathColor.h>
#include <Imath/ImathVec.h>

BSDL_ENTER_NAMESPACE

enum class ParamType : uint8_t {
    NONE = 0,
    VECTOR,
    INT,
    FLOAT,
    COLOR,
    STRING,
    CLOSURE
};

// This can easily be converted to OSL::ClosureParam
struct LobeParam {
    ParamType type;
    int offset;
    const char* key;
    int type_size;  // redundant?
};

template<typename T> struct ParamTypeOf {
    static BSDL_INLINE_METHOD constexpr ParamType get();
};

// clang-format off
template <> BSDL_INLINE_METHOD constexpr ParamType ParamTypeOf<Imath::V3f>::get()  { return ParamType::VECTOR;  }
template <> BSDL_INLINE_METHOD constexpr ParamType ParamTypeOf<int>::get()         { return ParamType::INT;     }
template <> BSDL_INLINE_METHOD constexpr ParamType ParamTypeOf<float>::get()       { return ParamType::FLOAT;   }
template <> BSDL_INLINE_METHOD constexpr ParamType ParamTypeOf<Imath::C3f>::get()  { return ParamType::COLOR;   }
template <> BSDL_INLINE_METHOD constexpr ParamType ParamTypeOf<const char*>::get() { return ParamType::STRING;  }
template <> BSDL_INLINE_METHOD constexpr ParamType ParamTypeOf<const void*>::get() { return ParamType::CLOSURE; }
// clang-format on

// Borrowed from https://gist.github.com/graphitemaster/494f21190bb2c63c5516 a
// handy way of grabbing the offset of a member in a legal way.
// Note the orignal fix doesn't work with multiple inheritance since &derived::member
// returns a pointer like base::*member for inherited members.
template<typename T> struct offset_in {
    template<typename T1, typename T2>
    static BSDL_INLINE_METHOD constexpr size_t of(T1 T2::*member)
    {
        // When a caller passes &type::foo as member, if foo is in a parent class it
        // will resolve as Parent::* (thanks C++) and T2 is NOT the intended struct we
        // want. This breaks the offset when using multiple inheritance.
        // To prevent the epic failure we cast the member pointer to be in T provided
        // explicitly.
        //constexpr T object{};
        auto&& object = T();  // Hoping to get a constexpr
        return size_t(&(object.*member)) - size_t(&object);
    }
};

template<typename D> struct LobeRegistry {
    using Data = D;

    struct Entry {
        static const unsigned MAX_PARAMS = 64;

        const char* name;
        LobeParam params[MAX_PARAMS];
    };
    template<typename T1, typename T2>
    static constexpr LobeParam param(T1 T2::*field, const char* key = nullptr)
    {
        return { ParamTypeOf<T1>::get(), (int)offset_in<Data>::of(field), key,
                 sizeof(T1) };
    }
    static LobeParam close()
    {
        return { ParamType::NONE, sizeof(Data), nullptr, alignof(Data) };
    }
};

// This is how to get an entry
// ClosureEntry<DiffuseLobe>()();

BSDL_LEAVE_NAMESPACE
