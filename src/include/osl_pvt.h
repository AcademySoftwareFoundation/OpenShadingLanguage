/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/

#ifndef OSL_PVT_H
#define OSL_PVT_H

#include "OpenImageIO/typedesc.h"


namespace OSL {
namespace pvt {


/// Kinds of shaders
///
enum ShaderType {
    ShadTypeUnknown, ShadTypeGeneric, ShadTypeSurface, 
    ShadTypeDisplacement, ShadTypeVolume, ShadTypeLight,
    ShadTypeLast
};



/// Kinds of symbols
///
enum SymType {
    SymTypeParam, SymTypeOutputParam,
    SymTypeLocal, SymTypeTemp, SymTypeGlobal, SymTypeConst,
    SymTypeFunction, SymTypeType
};



/// Light-weight way to describe types for the compiler -- simple types,
/// closures, or the ID of a structure.
class TypeSpec {
public:
    /// Default ctr of TypeSpec (unknown type)
    ///
    TypeSpec ()
        : m_simple(TypeDesc::UNKNOWN), m_structure(0), m_closure(false)
    { }

    /// Construct a TypeSpec that represents an ordinary simple type
    /// (including arrays of simple types).
    TypeSpec (TypeDesc simple)
        : m_simple(simple), m_structure(0), m_closure(false)
    { }

    /// Construct a TypeSpec representing a closure (pass closure=true)
    /// of a simple type.
    TypeSpec (TypeDesc simple, bool closure)
        : m_simple(simple), m_structure(0), m_closure(closure)
    { }

    /// Construct a TypeSpec describing a struct or array of structs,
    /// by supplying the struct name, structure id, and array length
    /// (if it's an array of structures).
    TypeSpec (const char *name, int structid, int arraylen=0)
        : m_simple(TypeDesc::UNKNOWN, arraylen), m_structure((short)structid),
          m_closure(false)
    { }

    /// Express the type as a string
    ///
    std::string string () const;

    /// Assignment of a simple TypeDesc to a full TypeSpec.
    ///
    const TypeSpec & operator= (const TypeDesc simple) {
        m_simple = simple;
        m_structure = 0;
        m_closure = false;
        return *this;
    }

    /// Are two TypeSpec's identical?
    ///
    bool operator== (const TypeSpec &x) const {
        return (m_simple == x.m_simple && m_structure == x.m_structure &&
                m_closure == x.m_closure);
    }
    /// Are two TypeSpec's different?
    ///
    bool operator!= (const TypeSpec &x) const { return ! (*this == x); }

    /// Return just the simple type underlying this TypeSpec -- only works
    /// reliable if it's not a struct, a struct will return an UNKNOWN type.
    TypeDesc simpletype () const { return m_simple; }

    /// Is this typespec a closure?  (N.B. if so, you can find out what
    /// kind of closure it is with simpletype()).
    bool is_closure () const { return m_closure; }

    /// Is this typespec a single structure?  Caveat: Returns false if
    /// it's an array of structs.  N.B. You can find out which struct
    /// with structure().
    bool is_structure () const { return m_structure > 0 && !is_array(); }

    /// Return the structure ID of this typespec, or 0 if it's not a
    /// struct.
    int structure () const { return m_structure; }

    /// Is this an array (either a simple array, or an array of structs)?
    ///
    bool is_array () const { return m_simple.arraylen != 0; }

    /// Returns the length of the array, or 0 if not an array.
    ///
    int arraylength () const { return m_simple.arraylen; }

    /// Alter this typespec to make it into an array of the given length
    /// (including 0 -> make it not be an array).  The basic type (not
    /// counting its array length) is unchanged.
    void make_array (int len) { m_simple.arraylen = len; }

    /// For an array, return the TypeSpec of an individual element of the
    /// array.  For a non-array, just return the type.
    TypeSpec elementtype () const { TypeSpec t; t.make_array (0); return t; }

    /// Is it an "aggregate" type, meaning a structure, closure, array,
    /// or a simple type that isn't a scalar (such as a vector/point)?
    bool is_aggregate () const {
        return !is_structure() && !is_closure() && 
               !is_array() && m_simple.aggregate != TypeDesc::SCALAR;
    }

    /// Return the aggregateness of the underlying simple type (SCALAR,
    /// VEC3, or MATRIX44).
    TypeDesc::AGGREGATE aggregate () const { return (TypeDesc::AGGREGATE)m_simple.aggregate; }

    /// Is it a simple scalar int?
    ///
    bool is_int () const {
        return m_simple == TypeDesc::TypeInt && !is_structure() && !is_closure();
    }

    /// Is it a simple scalar float?
    ///
    bool is_float () const {
        return m_simple == TypeDesc::TypeFloat && !is_structure() && !is_closure();
    }

    /// Is it a simple scalar float?
    ///
    bool is_color () const {
        return m_simple == TypeDesc::TypeColor && !is_structure() && !is_closure();
    }

    /// Is it a simple string?
    ///
    bool is_string () const {
        return m_simple == TypeDesc::TypeString && !is_structure() && !is_closure();
    }

    /// Is it a simple triple (color, point, vector, or normal)?
    ///
    bool is_triple () const {
        return ! is_structure() && ! is_closure() && 
            (m_simple == TypeDesc::TypeColor ||
             m_simple == TypeDesc::TypePoint ||
             m_simple == TypeDesc::TypeVector ||
             m_simple == TypeDesc::TypeNormal);
    }

    /// Is this a simple type based on floats (including color/vector/etc)?  
    /// This will return false for a closure or array (even if of floats)
    /// or struct.
    bool is_floatbased () const {
        return ! is_structure() && ! is_closure() && ! is_array() &&
            m_simple.basetype == TypeDesc::FLOAT;
    }

    /// Is it a simple numeric type (based on float or int, even if an
    /// aggregate)?  This is false for a closure or array (even if of
    /// an underlying numeric type) or struct.
    bool is_numeric () const {
        return ! is_structure() && ! is_closure() && ! is_array() &&
            (m_simple.basetype == TypeDesc::FLOAT || m_simple.basetype == TypeDesc::INT);
    }

    bool is_scalarnum () const {
        return is_numeric() && m_simple.aggregate == TypeDesc::SCALAR;
    }

    /// Is it a simple straight-up single int or float)?
    ///
    bool is_int_or_float () const { return is_scalarnum(); }

    /// Is it a simple vector-like triple (point, vector, or normal, but
    /// not an array or closure)?
    bool is_vectriple () const {
        return ! is_structure() && ! is_closure() && 
            (m_simple == TypeDesc::TypePoint ||
             m_simple == TypeDesc::TypeVector ||
             m_simple == TypeDesc::TypeNormal);
    }

    /// Is it based on a vector-like triple (point, vector, or normal)?
    /// (It's ok for it to be an array or closure.)
    bool is_vectriple_based () const {
        return ! is_structure() && 
            (m_simple.elementtype() == TypeDesc::TypePoint ||
             m_simple.elementtype() == TypeDesc::TypeVector ||
             m_simple.elementtype() == TypeDesc::TypeNormal);
    }

    /// Is it a simple matrix (but not an array or closure)?
    ///
    bool is_matrix () const {
        return ! is_structure() && ! is_closure() && 
            m_simple == TypeDesc::TypeMatrix;
    }

    /// Is it a color closure?
    ///
    bool is_color_closure () const {
        return is_closure() && (m_simple == TypeDesc::TypeColor);
    }

    /// Types are equivalent if they are identical, or if both are
    /// vector-like (and match their array-ness and closure-ness).
    friend bool equivalent (const TypeSpec &a, const TypeSpec &b) {
        return (a == b) || 
            (a.is_vectriple_based() && b.is_vectriple_based() &&
             a.is_closure() == b.is_closure() &&
             a.arraylength() == b.arraylength());
    }

    /// Is type b is assignable to a?  It is if they are the equivalent(),
    /// or if a is a float or float-aggregate and b is a float or int.
    friend bool assignable (const TypeSpec &a, const TypeSpec &b) {
        return equivalent (a, b) || 
            (a.is_floatbased() && (b.is_float() || b.is_int()));
    }

private:
    TypeDesc m_simple;     ///< Data if it's a simple type
    short m_structure;     ///< 0 is not a structure, >=1 for structure id
    bool  m_closure;       ///< Is it a closure? (m_simple also used)
};




}; // namespace OSL::pvt
}; // namespace OSL


#endif /* OSL_PVT_H */
