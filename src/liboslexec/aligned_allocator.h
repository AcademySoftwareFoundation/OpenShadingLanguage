/*
Copyright (c) 2017 Intel Corporation., et al.
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

/* NOTE:  Can be replaced with boost 1.56 boost::alligned_allocator
 * Current boost dependency is only boost 1.55, so in 2018 consider
 * removing this and using the boost version.
 * No expected performance difference, just one less file
 * NOTE: pointer_or_number and is_aligned appear useful and should be 
 * pulled into their own files
 */
#pragma once

#include <limits>
#include <type_traits>
#include <utility>

OSL_NAMESPACE_ENTER

namespace pvt {

// For converting pointers to numbers for alignment or other math operations
template<typename T = void>
union pointer_or_number {
    const T * pointer;
    size_t number;
};

template <size_t ByteAlignmentT, typename T>
inline bool is_aligned(const T *a_pointer)
{
    pointer_or_number<T> pon;
    pon.pointer = a_pointer;
    return (pon.number%ByteAlignmentT==0);
}

template <class ObjT>
class aligned_allocator
{
    template <class> friend class aligned_allocator;
    
    static constexpr int boundary = 64;
public:
    typedef ObjT value_type;
    typedef ObjT* pointer;
    typedef const ObjT* const_pointer;
    typedef ObjT& reference;
    typedef const ObjT& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::true_type propagate_on_container_move_assignment;
    typedef std::true_type is_always_equal;
    template< class OtherObjT >
    struct rebind
    {
        typedef aligned_allocator<OtherObjT> other;
    };


    pointer address( reference a_ref) const;
    const_pointer address( const_reference a_ref) const;

    
    pointer allocate(size_type a_count, std::allocator<void>::const_pointer a_hint = 0);
    void deallocate(pointer a_pointer, size_type a_count);

    size_type max_size() const;

    void construct(pointer a_pointer, const_reference a_value);

    void destroy(pointer a_pointer);

	// Required for C++11 compliance
	template< class OtherObjT, class... ArgsT >
	void construct(OtherObjT* a_pointer, ArgsT&&... args);
	template< class OtherObjT >
	void destroy(OtherObjT* a_pointer);
};

// Implementation


template <class ObjT>
typename aligned_allocator<ObjT>::pointer
aligned_allocator<ObjT>::address( reference a_ref) const
{
    return &a_ref;
}

template <class ObjT>
typename aligned_allocator<ObjT>::const_pointer
aligned_allocator<ObjT>::address( const_reference a_ref) const
{
    return &a_ref;
}

template <class ObjT>
typename aligned_allocator<ObjT>::pointer
aligned_allocator<ObjT>::allocate(size_type a_count, std::allocator<void>::const_pointer /*a_hint*/)
{
    size_t byte_count = sizeof(ObjT)*a_count;
    		
    size_t padded_byte_count = byte_count + boundary;
    void * base_pointer = ::malloc(padded_byte_count);
	if (base_pointer == nullptr)
	{
		return nullptr;
	}
	assert(is_aligned<sizeof(int)>(base_pointer));

    pointer_or_number<> pon;
    pon.pointer = base_pointer;

    int byte_count_to_cacheline_boundary = static_cast<int>(pon.number%boundary);
    // Check that we will always pad enough to store an int before the data pointer
    assert((boundary - byte_count_to_cacheline_boundary)>static_cast<int>(sizeof(int)));

    void * data_pointer = reinterpret_cast<unsigned char *>(base_pointer) + boundary - byte_count_to_cacheline_boundary;
    // Store how many bytes we skipped in the 1st byte before the data_pointer,
    // we will always skip 4 to boundary bytes, so there should always be room
    *(reinterpret_cast<int *>(reinterpret_cast<unsigned char *>(data_pointer) - sizeof(int))) = byte_count_to_cacheline_boundary;
    return reinterpret_cast<pointer>(data_pointer);   
}

template <class ObjT>
void
aligned_allocator<ObjT>::deallocate(pointer a_pointer, size_type /*a_count*/)
{
    int byte_count_to_cacheline_boundary = *(reinterpret_cast<int *>(reinterpret_cast<unsigned char *>(a_pointer) - sizeof(int)));
    void * base_pointer = reinterpret_cast<unsigned char *>(a_pointer) - boundary + byte_count_to_cacheline_boundary;
    ::free(base_pointer);    
}

template <class ObjT>
typename aligned_allocator<ObjT>::size_type
aligned_allocator<ObjT>::max_size() const
{
    // Overly optimistic, but sure an allocation greater than this would fail
    return std::numeric_limits<size_type>::max()/sizeof(ObjT);
}

template <class ObjT>
void
aligned_allocator<ObjT>::construct(pointer a_pointer, const_reference a_value)
{
    ::new((void *)(a_pointer)) value_type(a_value);
}

template <class ObjT>
void
aligned_allocator<ObjT>::destroy(pointer a_pointer)
{
    a_pointer->~ObjT();
}


template <class ObjT>
template< class OtherObjT, class... ArgsT >
void
aligned_allocator<ObjT>::construct(OtherObjT* a_pointer, ArgsT&&... args)
{
	::new((void *)(a_pointer)) OtherObjT(std::forward<ArgsT>(args)...);
}

template <class ObjT>
template< class OtherObjT >
void
aligned_allocator<ObjT>::destroy(OtherObjT* a_pointer)
{
	a_pointer->~OtherObjT();
}

}; // namespace pvt


OSL_NAMESPACE_EXIT
