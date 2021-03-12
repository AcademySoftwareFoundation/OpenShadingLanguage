// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <OpenImageIO/oiioversion.h>
#include <OpenImageIO/detail/farmhash.h>

namespace pvt {

OIIO_CONSTEXPR14 inline size_t pvtstrlen(const char *s) {
    if (s == nullptr)
        return 0;
    size_t  len = 0;
    while (s[len] != 0)
        len++;
    return len;
}

}

namespace UStringHash {
OIIO_CONSTEXPR14 inline size_t Hash(const char* s)
{
  size_t len = pvt::pvtstrlen(s);

  return len ? OIIO::farmhash::inlined::Hash(s, len)
             : 0;
}
}

