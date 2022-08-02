// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OpenImageIO/detail/farmhash.h>
#include <OpenImageIO/oiioversion.h>

namespace pvt {

OIIO_CONSTEXPR14 inline size_t
pvtstrlen(const char* s)
{
    if (s == nullptr)
        return 0;
    size_t len = 0;
    while (s[len] != 0)
        len++;
    return len;
}

}  // namespace pvt

namespace UStringHash {
OIIO_CONSTEXPR14 inline size_t
Hash(const char* s)
{
    size_t len = pvt::pvtstrlen(s);

    return len ? OIIO::farmhash::inlined::Hash(s, len) : 0;
}
}  // namespace UStringHash
