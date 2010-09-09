/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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

#ifndef SMALLVEC_H
#define SMALLVEC_H

#include <cstring>
#include "oslconfig.h"
#include "OpenImageIO/dassert.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

/// A small helper class that emulates std::vector<> except that it
/// has a small fixed-size allocation (of StaticSize elements) to
/// avoid heap allocation. This is similar in spirit to LLVM's
/// SmallVector but only works properly for POD (uses memcpy), doesn't
/// shrink and doesn't really provide any safety/iterators.
template<typename T, int StaticSize>
class SmallVec {
public:
    SmallVec() : current_buffer(static_alloc), dynamic_alloc(NULL), num_elements(0), num_dynamic_slots(0) {
    }
    ~SmallVec() {
        if (dynamic_alloc != NULL) free(dynamic_alloc);
    }

    SmallVec(const SmallVec& other) : current_buffer(static_alloc), dynamic_alloc(NULL), num_elements(0), num_dynamic_slots(0) {
        reserve(other.size(), false);
        memcpy(current_buffer, other.current_buffer, sizeof(T) * other.size());
        num_elements = other.size();
    }

    SmallVec& operator=(const SmallVec& other) {
        reserve(other.size(), false);
        memcpy(current_buffer, other.current_buffer, sizeof(T) * other.size());
        num_elements = other.size();
        return *this;
    }

    void clear() {
        num_elements = 0;
        current_buffer = static_alloc;
    }

    void reserve_dynamic(int new_size)
    {
        if (new_size > num_dynamic_slots) {
            int available = StaticSize > num_dynamic_slots ? StaticSize : num_dynamic_slots;
            int desired_size = (new_size > 2 * available) ? new_size : 2 * available;
            dynamic_alloc = (T*) realloc(dynamic_alloc, sizeof(T)*desired_size); //new T[desired_size];
            num_dynamic_slots = desired_size;
            if (current_buffer != static_alloc)
                current_buffer = dynamic_alloc;
        }
    }

    void reserve(size_t new_size, bool preserve_data = true) {
        if (!new_size) {
            clear();
        } else if (current_buffer == dynamic_alloc) {
            if (new_size <= StaticSize) {
                if (preserve_data && num_elements)
                    memcpy(static_alloc, dynamic_alloc,
                           ((size_t)num_elements < new_size ? (size_t)num_elements : new_size) * sizeof(T));
                current_buffer = static_alloc;
            } else {
                reserve_dynamic(new_size);
            }
        } else if (new_size > StaticSize) {
            reserve_dynamic(new_size);
            if (preserve_data && num_elements)
                memcpy(dynamic_alloc, static_alloc, sizeof(T) * num_elements);
            current_buffer = dynamic_alloc;
        }
    }

    void resize(size_t new_size) {
        reserve(new_size);
        num_elements = new_size;
    }

    size_t size() const {
        return num_elements;
    }

    void push_back(const T& val) {
        reserve((size_t)num_elements + 1);
        current_buffer[num_elements++] = val;
    }

    T& operator[](int index) {
        DASSERT((current_buffer ==  static_alloc && index < StaticSize) ||
                (current_buffer ==  dynamic_alloc && index < num_dynamic_slots));
        return current_buffer[index];
    }

    const T& operator[](int index) const {
        DASSERT((current_buffer ==  static_alloc && index < StaticSize) ||
                (current_buffer ==  dynamic_alloc && index < num_dynamic_slots));
        return current_buffer[index];
    }

    T& back() {
        DASSERT(num_elements > 0);
        return current_buffer[num_elements - 1];
    }

    const T& back() const {
        DASSERT(num_elements > 0);
        return current_buffer[num_elements - 1];
    }

    size_t get_memory_usage() const {
      return sizeof(SmallVec) + sizeof(T) * num_dynamic_slots;
    }

    T* current_buffer;
    T* dynamic_alloc;
    int num_elements;
    int num_dynamic_slots;
    T static_alloc[StaticSize];
};

}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif
