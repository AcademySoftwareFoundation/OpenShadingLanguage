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

#ifndef OSLCLOSURE_H
#define OSLCLOSURE_H

#include <cstring>
#include <OpenImageIO/ustring.h>
#include "oslconfig.h"

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
    enum ClosureType { COMPONENT, MUL, ADD };

    ClosureType type;
};



/// ClosureComponent is a subclass of ClosureColor that holds the ID and
/// parameter data for a single primitive closure component (such as
/// diffuse, translucent, etc.).  The declaration leaves 4 bytes for
/// parameter data (mem), but it's expected that the structure be
/// allocated with enough space to house all the parameter data for
/// whatever type of custom primitive component it actually is.
struct OSLEXECPUBLIC ClosureComponent : public ClosureColor
{
    struct Attr
    {
        ustring   key;
        union {
            int     integer;
            float   flt;
            float   triple[3];  // This will fake Color3 and Vec3 which C++ doesn't allow in unions
            void   *str;        // And this fakes a ustring (not allowed in unions either)
        }         value;

        // This members are just to avoid having to typecast all the time
        int           & integer()       { return value.integer; }
        const int     & integer() const { return value.integer; }
        float         & flt()           { return value.flt; }
        const float   & flt()     const { return value.flt; }
        Color3        & color()         { return *(Color3*)       raw_data(); }
        const Color3  & color()   const { return *(const Color3*) raw_data(); }
        Vec3          & vector()        { return *(Vec3*)         raw_data(); }
        const Vec3    & vector()  const { return *(const Vec3*)   raw_data(); }
        ustring       & str()           { return *(ustring*)      raw_data(); }
        const ustring & str()     const { return *(const ustring*)raw_data(); }

    private:
        char*       raw_data()       { return reinterpret_cast<char*>(&value); }
        const char* raw_data() const { return reinterpret_cast<const char*>(&value); }
    };

    int    id;       ///< Id of the component
    int    size;     ///< Memory used by the primitive
    int    nattrs;   ///< Number of keyword attributes
    Vec3   w;        ///< Weight of this component
    char   mem[4];   ///< Memory for the primitive
                     ///  4 is the minimum, allocation
                     ///  will be scaled to requirements
                     ///  of the primitive

    /// Handy method for getting the parameter memory as a void*.
    ///
    void *data () { return &mem; }
    const void *data () const { return &mem; }
    /// Attributes are always allocated at the end of the data block
    Attr *attrs() { return (Attr *)((char *)data() + size); }
    const Attr *attrs() const { return (Attr *)((char *)data() + size); }
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

#endif /* OSLCLOSURE_H */
