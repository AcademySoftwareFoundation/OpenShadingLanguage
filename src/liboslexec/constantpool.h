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

#ifndef CONSTANTPOOL_H
#define CONSTANTPOOL_H

#include <vector>
#include <list>
#include <boost/foreach.hpp>

#include "OpenImageIO/thread.h"
#ifdef OIIO_NAMESPACE
namespace OIIO = OIIO_NAMESPACE;
using OIIO::mutex;
using OIIO::lock_guard;
#endif

using namespace OSL;
using namespace OSL::pvt;


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
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
        lock_guard lock (m_mutex);
        // Check each block in the block list to see if it has enough space
        BOOST_FOREACH (block_t &block, m_block_list) {
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
    mutex m_mutex;     ///< Thread-safe lock
};



}; // namespace OSL::pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif

#endif /* CONSTANTPOOL_H */
