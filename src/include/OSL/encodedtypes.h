// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>



enum class EncodedType : uint8_t
{
    kUstringHash,
    kInt32,
    kFloat,
    kDouble, // TODO: remove kDouble
};
template<typename T>
struct TypeEncoder; //Undefined on purpose; all types must have a specialization


namespace {  
        // MAYBE DONT NEED THIS
// ensure each compilation unit has its own lookup table by putting it in unnamed namespace
constexpr uint32_t SizeByEncodedType[] = {
   sizeof(OSL::ustringhash),
   sizeof(int32_t),
   sizeof(float),
   sizeof(double),
   };
};

template<>
struct TypeEncoder<OSL::ustringhash>
{
        using DataType = OSL::ustringhash;
        static constexpr EncodedType value = EncodedType::kUstringHash;
};

template<>
struct TypeEncoder<OSL::ustring>
{
        using DataType = OSL::ustringhash;
        static constexpr EncodedType value = EncodedType::kUstringHash;
};

template<>
struct TypeEncoder<int32_t>
{
        using DataType = int32_t;
        static constexpr EncodedType value = EncodedType::kInt32;
};

template<>
struct TypeEncoder<float>
{
        using DataType = float;
        static constexpr EncodedType value = EncodedType::kFloat;
};

template<>
struct TypeEncoder<double>
{
        using DataType = double;
        static constexpr EncodedType value = EncodedType::kDouble;
};
