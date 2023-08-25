// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once


#include <OSL/oslconfig.h>


OSL_NAMESPACE_ENTER

// TODO: When minimum required C++ reaches C++17, replace with std::variant.
template<typename TBuiltinArg> class ArgVariant {
public:
    enum class Type {
        Unspecified = 0,
        Builtin,
        Bool,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Float,
        Double,
        Pointer,
        UString,
        UStringHash,
    };

private:
    union {
        TBuiltinArg m_builtin;
        bool m_bool;
        int8_t m_int8;
        int16_t m_int16;
        int32_t m_int32;
        int64_t m_int64;
        uint8_t m_uint8;
        uint16_t m_uint16;
        uint32_t m_uint32;
        uint64_t m_uint64;
        float m_float;
        double m_double;
        void* m_ptr;
        ustring m_ustring;
        ustringhash m_ustringhash;
    };
    Type m_type;

public:
    ArgVariant() : m_type(Type::Unspecified) {}
    ArgVariant(TBuiltinArg arg) : m_builtin(arg), m_type(Type::Builtin) {}
    ArgVariant(bool arg) : m_bool(arg), m_type(Type::Bool) {}
    ArgVariant(int8_t arg) : m_int8(arg), m_type(Type::Int8) {}
    ArgVariant(int16_t arg) : m_int16(arg), m_type(Type::Int16) {}
    ArgVariant(int32_t arg) : m_int32(arg), m_type(Type::Int32) {}
    ArgVariant(int64_t arg) : m_int64(arg), m_type(Type::Int64) {}
    ArgVariant(uint8_t arg) : m_uint8(arg), m_type(Type::UInt8) {}
    ArgVariant(uint16_t arg) : m_uint16(arg), m_type(Type::UInt16) {}
    ArgVariant(uint32_t arg) : m_uint32(arg), m_type(Type::UInt32) {}
    ArgVariant(uint64_t arg) : m_uint64(arg), m_type(Type::UInt64) {}
    ArgVariant(float arg) : m_float(arg), m_type(Type::Float) {}
    ArgVariant(double arg) : m_double(arg), m_type(Type::Double) {}
    ArgVariant(void* arg) : m_ptr(arg), m_type(Type::Pointer) {}
    ArgVariant(ustring arg) : m_ustring(arg), m_type(Type::UString) {}
    ArgVariant(ustringhash arg) : m_ustringhash(arg), m_type(Type::UStringHash)
    {
    }

    ArgVariant(const ArgVariant& other)
    {
        memcpy((void*)this, &other, sizeof(ArgVariant));
    }
    ArgVariant(ArgVariant&& other)
    {
        memcpy((void*)this, &other, sizeof(ArgVariant));
    }

    ~ArgVariant() {}

    ArgVariant& operator=(const ArgVariant& other)
    {
        memcpy((void*)this, &other, sizeof(ArgVariant));
        return *this;
    }
    ArgVariant& operator=(ArgVariant&& other)
    {
        memcpy((void*)this, &other, sizeof(ArgVariant));
        return *this;
    }

    Type type() const { return m_type; }

    template<typename T> bool is_holding() const
    {
        if (std::is_same<T, TBuiltinArg>::value)
            return m_type == Type::Builtin;
        if (std::is_same<T, bool>::value)
            return m_type == Type::Bool;
        if (std::is_same<T, int8_t>::value)
            return m_type == Type::Int8;
        if (std::is_same<T, int16_t>::value)
            return m_type == Type::Int16;
        if (std::is_same<T, int32_t>::value)
            return m_type == Type::Int32;
        if (std::is_same<T, int64_t>::value)
            return m_type == Type::Int64;
        if (std::is_same<T, uint8_t>::value)
            return m_type == Type::UInt8;
        if (std::is_same<T, uint16_t>::value)
            return m_type == Type::UInt16;
        if (std::is_same<T, uint32_t>::value)
            return m_type == Type::UInt32;
        if (std::is_same<T, uint64_t>::value)
            return m_type == Type::UInt64;
        if (std::is_same<T, float>::value)
            return m_type == Type::Float;
        if (std::is_same<T, double>::value)
            return m_type == Type::Double;
        if (std::is_same<T, void*>::value)
            return m_type == Type::Pointer;
        if (std::is_same<T, ustring>::value)
            return m_type == Type::UString;
        if (std::is_same<T, ustringhash>::value)
            return m_type == Type::UStringHash;

        return false;
    }

    TBuiltinArg get_builtin() const
    {
        OSL_DASSERT(is_holding<TBuiltinArg>());
        return m_builtin;
    }

    bool get_bool() const
    {
        OSL_DASSERT(is_holding<bool>());
        return m_bool;
    }

    int8_t get_int8() const
    {
        OSL_DASSERT(is_holding<int8_t>());
        return m_int8;
    }

    int16_t get_int16() const
    {
        OSL_DASSERT(is_holding<int16_t>());
        return m_int16;
    }

    int32_t get_int32() const
    {
        OSL_DASSERT(is_holding<int32_t>());
        return m_int32;
    }

    int64_t get_int64() const
    {
        OSL_DASSERT(is_holding<int64_t>());
        return m_int64;
    }

    uint8_t get_uint8() const
    {
        OSL_DASSERT(is_holding<uint8_t>());
        return m_uint8;
    }

    uint16_t get_uint16() const
    {
        OSL_DASSERT(is_holding<uint16_t>());
        return m_uint16;
    }

    uint32_t get_uint32() const
    {
        OSL_DASSERT(is_holding<uint32_t>());
        return m_uint32;
    }

    uint64_t get_uint64() const
    {
        OSL_DASSERT(is_holding<uint64_t>());
        return m_uint64;
    }

    float get_float() const
    {
        OSL_DASSERT(is_holding<float>());
        return m_float;
    }

    double get_double() const
    {
        OSL_DASSERT(is_holding<double>());
        return m_double;
    }

    void* get_ptr() const
    {
        OSL_DASSERT(is_holding<void*>());
        return m_ptr;
    }

    ustring get_ustring() const
    {
        OSL_DASSERT(is_holding<ustring>());
        return m_ustring;
    }

    ustringhash get_ustringhash() const
    {
        OSL_DASSERT(is_holding<ustringhash>());
        return m_ustringhash;
    }
};


// The FunctionSpec is never in device code, so it can be a bit more
// complex and have dynamic allocations.  Although we do want to
// use an interface and not allow direct data member access
template<typename TArg> class FunctionSpec {
    ustring m_function_name;
    std::vector<TArg> m_args;

public:
    template<typename... ArgListT>
    void set(ustring function_name, ArgListT... args)
    {
        m_function_name = function_name;
        m_args.clear();
        m_args.insert(m_args.end(),
                      std::initializer_list<TArg> { TArg { args }... });
    }

    const ustring& function_name() const { return m_function_name; }
    size_t arg_count() const { return m_args.size(); }
    const TArg& arg(size_t i) const
    {
        OSL_DASSERT(i < m_args.size());
        return m_args[i];
    }
};


OSL_NAMESPACE_EXIT
