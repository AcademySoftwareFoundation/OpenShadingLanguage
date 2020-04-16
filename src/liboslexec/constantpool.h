// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <vector>
#include <list>

#include <OpenImageIO/thread.h>

using namespace OSL;
using namespace OSL::pvt;


OSL_NAMESPACE_ENTER

namespace pvt {


/// A ConstantPool<T> is a way to allocate room for a small number of
/// T's at a time, such that the memory allocated will NEVER change its
/// address or be deallocated until the entire ConstantPool is
/// destroyed.  Allocating from the pool is completely thread-safe.
///
/// It is implemented as a linked list of memory blocks.  A request for
/// a new allocation tries to fit it in one of the allocated blocks, but
/// if it won't fit anywhere, it makes a new block and adds it to the
/// head of the list.
template<class T>
class ConstantPool {
public:
    /// Allocate a new pool of T's.  The quanta, if supplied, is the
    /// number of T's to malloc at a time.
    ConstantPool (size_t quanta = 1000000) : m_quanta(quanta), m_total(0) { }

    ~ConstantPool () { }

    /// Allocate space enough for n T's, and return a pointer to the
    /// start of that space.
    T * alloc (size_t n) {
        OIIO::lock_guard lock (m_mutex);
        // Check each block in the block list to see if it has enough space
        for (auto&& block : m_block_list) {
            size_t s = block.size();
            if ((s+n) <= block.capacity()) {
                // Enough space in this block.  Use it.
                block.resize (s+n);
                return &block[s];
            }
        }
        // If we got here, there were no mini-blocks in the list with enough
        // space.  Make a new one.
        m_block_list.push_front (block_t());
        block_t &block (m_block_list.front());
        size_t s = std::max (m_quanta, n);
        block.reserve (s);
        m_total += s * sizeof(T);
        block.resize (n);
        return &block[0];
    }

private:
    typedef std::vector<T> block_t;   ///< Type of block
    std::list<block_t> m_block_list;  ///< List of memory blocks
    size_t m_quanta;   ///< How big each memory block is (in T's, not bytes)
    size_t m_total;    ///< Total memory allocated (bytes!)
    OIIO::mutex m_mutex;  ///< Thread-safe lock
};



}; // namespace OSL::pvt
OSL_NAMESPACE_EXIT
