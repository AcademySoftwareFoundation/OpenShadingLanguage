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

#pragma once

#include <cstring>
#include <OSL/oslconfig.h>

OSL_NAMESPACE_ENTER

/// Labels for light walks
///
/// This is the leftover of a class which used to hold all the labels
/// Now just acts as a little namespace for the basic definitions
///
/// NOTE: you still can assign these labels as keyword arguments to
///       closures. But we have removed the old labels array in the
///       primitives.
class OSLEXECPUBLIC Labels {
public:

    static const ustring NONE;
    // Event type
    static const ustring CAMERA;
    static const ustring LIGHT;
    static const ustring BACKGROUND;
    static const ustring TRANSMIT;
    static const ustring REFLECT;
    static const ustring VOLUME;
    static const ustring OBJECT;
    // Scattering
    static const ustring DIFFUSE;  // typical 2PI hemisphere
    static const ustring GLOSSY;   // blurry reflections and transmissions
    static const ustring SINGULAR; // perfect mirrors and glass
    static const ustring STRAIGHT; // Special case for transparent shadows

    static const ustring STOP;     // end of a surface description

};

// Forward declarations
struct ClosureComponent;
struct ClosureMul;
struct ClosureAdd;

/// ClosureColor is the base class for a lightweight tree representation
/// of OSL closures for the sake of the executing OSL shader.
///
/// Remember that a closure color really just boils down to a flat list
/// of weighted closure primitive components (such as diffuse,
/// transparent, etc.).  But it's expensive to construct these
/// dynamically as we execute the shader, so instead of maintaining a
/// properly flattened list as we go, we just manipulate a very
/// lightweight data structure that looks like a tree, where leaf nodes
/// are a single primitive component, and internal nodes of the tree are
/// are either 'add' (two closures) or 'mul' (of a closure with a
/// weight) and just reference their operands by pointers.
/// 
/// We are extremely careful to make these classes resemble POD (plain
/// old data) so they can be easily "placed" anywhere, including a memory
/// pool.  So no virtual functions!
///
/// The base class ClosureColor just provides the type, and it's
/// definitely one of the three kinds of subclasses: ClosureComponent,
/// ClosureMul, ClosureAdd.
struct OSLEXECPUBLIC ClosureColor {
    enum ClosureID { COMPONENT_BASE_ID = 0, MUL = -1, ADD = -2 };

    int id;

    OSL_HOSTDEVICE const ClosureComponent* as_comp() const {
        DASSERT(id >= COMPONENT_BASE_ID);
        return reinterpret_cast<const ClosureComponent*>(this);
    }

    OSL_HOSTDEVICE const ClosureMul* as_mul() const {
        DASSERT(id == MUL);
        return reinterpret_cast<const ClosureMul*>(this);
    }

    OSL_HOSTDEVICE const ClosureAdd* as_add() const {
        DASSERT(id == ADD);
        return reinterpret_cast<const ClosureAdd*>(this);
    }
};



/// ClosureComponent is a subclass of ClosureColor that holds the ID and
/// parameter data for a single primitive closure component (such as
/// diffuse, translucent, etc.).
///
/// ClosureComponent itself takes up 16 bytes, and its allocation will be
/// scaled to add parameters after the end of the struct. Alignment is
/// set to 16 bytes so that 64 bit pointers and 128 bit SSE types in user
/// structs have the required alignment.
#ifdef __CUDACC__
/// Notice in the OptiX implementation we align this to 8 bytes
/// so that it matches the alignment of the memory pools.
struct OSLEXECPUBLIC OSL_ALIGNAS(8) ClosureComponent : public ClosureColor
#else
struct OSLEXECPUBLIC OSL_ALIGNAS(16) ClosureComponent : public ClosureColor
#endif
{
    Vec3 w;                     ///< Weight of this component

    /// Handy method for getting the parameter memory as a void*.
    OSL_HOSTDEVICE void *data () { return (char*)(this + 1); }
    OSL_HOSTDEVICE const void *data () const { return (const char*)(this + 1); }

    /// Handy methods for extracting the underlying parameters as a struct
    template <typename T>
    OSL_HOSTDEVICE const T* as() const { return reinterpret_cast<const T*>(data()); }

    template <typename T>
    OSL_HOSTDEVICE T* as() { return reinterpret_cast<const T*>(data()); }
};


/// ClosureMul is a subclass of ClosureColor that provides a lightweight
/// way to represent a closure multiplied by a scalar or color weight.
struct OSLEXECPUBLIC ClosureMul : public ClosureColor
{
    Color3 weight;
    const ClosureColor *closure;
};


/// ClosureAdd is a subclass of ClosureColor that provides a lightweight
/// way to represent a closure that is a sum of two other closures.
struct OSLEXECPUBLIC ClosureAdd : public ClosureColor
{
    const ClosureColor *closureA;
    const ClosureColor *closureB;
};

OSL_NAMESPACE_EXIT
