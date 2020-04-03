/*
Copyright (c) 2009-2018 Sony Pictures Imageworks Inc., et al.
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

#include <OSL/oslversion.h>

#ifdef OSL_TBB_SCALABLE_MALLOC
    #include <tbb/scalable_allocator.h>
    #if (__TBB_ALLOCATOR_CONSTRUCT_VARIADIC == 0)
        #error (__TBB_ALLOCATOR_CONSTRUCT_VARIADIC == 0) Minimum language/library requirements not met.  Try defining TBB_USE_GLIBCXX_VERSION to the actual GCC version on the system (ICC and Clang may not properly identify it by themselves).  e.g. for gcc6.3: -DTBB_USE_GLIBCXX_VERSION=60301
    #endif
#endif

#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <vector>

#include <OpenImageIO/span.h>

OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt

#ifdef OSL_TBB_SCALABLE_MALLOC
    // The tbb::scalable_allocator<T> manages
    // per thread pools of memory so smaller allocations will reuse
    // previously freed memory from the same thread.  This reuse avoids
    // multiple threads contending for OS resources and avoids the
    // overhead of going to the OS for memory requests when possible.
    // Intent is to reduce overhead incurred in the OSL runtime
    // compilation/optimization stages as much as possible,
    // especially when used in a multithreaded environment.
    template <typename T>
    using CustomAllocator = tbb::scalable_allocator<T>;
#else
    template <typename T>
    using CustomAllocator = std::allocator<T>;
#endif

template <
    typename DataT,
    typename AllocatorT=CustomAllocator<DataT>
> using vector = std::vector<DataT, AllocatorT>;

// TODO: OIIO::cspan contructor only accepts std::vector<T>, it needs to
// be expanded to handle non-default allocators.
// In the mean time, provide make helpers to create spans
template <typename T, typename AllocatorT>
OIIO::cspan<T> make_cspan(const std::vector<T, AllocatorT> &dynamic_array) {
    auto size = static_cast<typename OIIO::cspan<T>::index_type>(dynamic_array.size());
	return {dynamic_array.data(), size};
}
template <typename T, typename AllocatorT>
OIIO::span<T> make_span(std::vector<T, AllocatorT> &dynamic_array) {
    auto size = static_cast<typename OIIO::span<T>::index_type>(dynamic_array.size());
	return {dynamic_array.data(), size};
}
// Deduce type of cspan based on type of pointer
// TODO:  Add c++17 conditional support for CTAD in OIIO::cspan
template <typename T>
OIIO::cspan<T> make_cspan(const T * array_pointer, size_t length) {
    auto size = static_cast<typename OIIO::cspan<T>::index_type>(length);
	return {array_pointer, size};
}

template <
    typename KeyT,
    typename DataT,
    typename HashT = std::hash<KeyT>,
    typename KeyEqualT = std::equal_to<KeyT>,
    typename AllocatorT=CustomAllocator< std::pair<const KeyT, DataT> >
> using unordered_map = std::unordered_map<KeyT, DataT, HashT, KeyEqualT, AllocatorT >;

template <
    typename KeyT,
    typename HashT = std::hash<KeyT>,
    typename KeyEqualT = std::equal_to<KeyT>,
    typename AllocatorT=CustomAllocator<KeyT>
> using unordered_set = std::unordered_set<KeyT, HashT, KeyEqualT, AllocatorT >;

template<
    class KeyT,
    class DataT,
    class CompareT = std::less<KeyT>,
    class AllocatorT = CustomAllocator< std::pair<const KeyT, DataT> >
> using map = std::map<KeyT, DataT, CompareT, AllocatorT>;

template<
    class KeyT,
    class CompareT = std::less<KeyT>,
    class AllocatorT = CustomAllocator<KeyT>
> using set = std::set<KeyT, CompareT, AllocatorT>;

template<
    class DataT,
    class AllocatorT = CustomAllocator<DataT>
> using list = std::list<DataT, AllocatorT>;

template<
    class DataT,
    class ContainerT = std::deque<DataT, CustomAllocator<DataT>>
> using stack = std::stack<DataT, ContainerT>;

}; // namespace pvt
OSL_NAMESPACE_EXIT
