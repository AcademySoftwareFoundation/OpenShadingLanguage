// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/thread.h>

#include "oslexec_pvt.h"



OSL_NAMESPACE_ENTER

namespace pvt {  // OSL::pvt



std::vector<std::shared_ptr<StructSpec>>&
TypeSpec::struct_list()
{
    static std::vector<std::shared_ptr<StructSpec>> m_structs;
    return m_structs;
}



TypeSpec::TypeSpec(const char* name, int structid, int arraylen)
    : m_simple(TypeDesc::UNKNOWN, arraylen)
    , m_structure((short)structid)
    , m_closure(false)
{
    if (m_structure == 0)
        m_structure = structure_id(name, true);
}



std::string
TypeSpec::string() const
{
    std::string str;
    if (is_closure() || is_closure_array()) {
        str += "closure color";
        if (is_unsized_array())
            str += "[]";
        else if (arraylength() > 0)
            str += Strutil::fmt::format("[{}]", arraylength());
    } else if (structure() > 0) {
        StructSpec* ss = structspec();
        if (ss)
            str += Strutil::fmt::format("struct {}", structspec()->name());
        else
            str += Strutil::fmt::format("struct {}", structure());
        if (is_unsized_array())
            str += "[]";
        else if (arraylength() > 0)
            str += Strutil::fmt::format("[{}]", arraylength());
    } else {
        str += simpletype().c_str();
    }
    return str;
}



const char*
TypeSpec::c_str() const
{
    ustring s(this->string());
    return s.c_str();
}


const char*
TypeSpec::type_c_str() const
{
    if (is_structure())
        return ustring::fmtformat("struct {}", structspec()->name()).c_str();
    else
        return c_str();
}


int
TypeSpec::structure_id(const char* name, bool add)
{
    std::vector<std::shared_ptr<StructSpec>>& m_structs(struct_list());
    ustring n(name);
    for (int i = (int)m_structs.size() - 1; i > 0; --i) {
        if (m_structs[i] && m_structs[i]->name() == n)
            return i;
    }
    if (add) {
        if (m_structs.size() >= 0x8000) {
            OSL_ASSERT(0 && "more struct id's than fit in a short!");
            return 0;
        }
        int id = new_struct(new StructSpec(n, 0));
        return id;
    }
    return 0;  // Not found, not added
}



int
TypeSpec::new_struct(StructSpec* n)
{
    std::vector<std::shared_ptr<StructSpec>>& m_structs(struct_list());
    if (m_structs.size() == 0)
        m_structs.resize(1);  // Allocate an empty one
    m_structs.push_back(std::shared_ptr<StructSpec>(n));
    return (int)m_structs.size() - 1;
}

TypeSpec
TypeSpec::type_from_code(const char* code, int* advance)
{
    TypeSpec t;
    int i = 0;
    switch (code[i]) {
    case 'i': t = TypeInt; break;
    case 'f': t = TypeFloat; break;
    case 'c': t = TypeColor; break;
    case 'p': t = TypePoint; break;
    case 'v': t = TypeVector; break;
    case 'n': t = TypeNormal; break;
    case 'm': t = TypeMatrix; break;
    case 's': t = TypeString; break;
    case 'h': t = OSL::TypeUInt64; break;  // ustringhash_pod
    case 'x': t = TypeDesc(TypeDesc::NONE); break;
    case 'X': t = TypeDesc(TypeDesc::PTR); break;
    case 'L': t = TypeDesc(TypeDesc::LONGLONG); break;
    case 'C':  // color closure
        t = TypeSpec(TypeColor, true);
        break;
    case 'S':  // structure
        // Following the 'S' is the numeric structure ID
        t = TypeSpec("struct", atoi(code + i + 1));
        // Skip to the last digit
        while (isdigit(code[i + 1]))
            ++i;
        break;
    case '?': break;  // anything will match, so keep 'UNKNOWN'
    case '*': break;  // anything will match, so keep 'UNKNOWN'
    case '.': break;  // anything will match, so keep 'UNKNOWN'
    default:
        OSL_DASSERT_MSG(0, "Don't know how to decode type code '%d'",
                        (int)code[0]);
        if (advance)
            *advance = 1;
        return TypeSpec();
    }
    ++i;

    if (code[i] == '[') {
        ++i;
        t.make_array(-1);  // signal arrayness, unknown length
        if (isdigit(code[i]) || code[i] == ']') {
            if (isdigit(code[i]))
                t.make_array(atoi(code + i));
            while (isdigit(code[i]))
                ++i;
            if (code[i] == ']')
                ++i;
        }
    }

    if (advance)
        *advance = i;
    return t;
}

std::string
TypeSpec::typelist_from_code(const char* code)
{
    std::string ret;
    while (*code) {
        // Handle some special cases
        int advance = 1;
        if (ret.length())
            ret += ", ";
        if (*code == '.') {
            ret += "...";
        } else if (*code == 'T') {
            ret += "...";
        } else if (*code == '?') {
            ret += "<any>";
        } else {
            TypeSpec t = TypeSpec::type_from_code(code, &advance);
            ret += t.type_c_str();
        }
        code += advance;
        if (*code == '[') {
            ret += "[]";
            ++code;
            while (isdigit(*code))
                ++code;
            if (*code == ']')
                ++code;
        }
    }

    return ret;
}



std::string
TypeSpec::code_from_type() const
{
    std::string out;
    TypeDesc elem = elementtype().simpletype();
    if (is_structure() || is_structure_array()) {
        out = Strutil::fmt::format("S{}", structure());
    } else if (is_closure() || is_closure_array()) {
        out = 'C';
    } else {
        if (elem == TypeInt)
            out = 'i';
        else if (elem == TypeFloat)
            out = 'f';
        else if (elem == TypeColor)
            out = 'c';
        else if (elem == TypePoint)
            out = 'p';
        else if (elem == TypeVector)
            out = 'v';
        else if (elem == TypeNormal)
            out = 'n';
        else if (elem == TypeMatrix)
            out = 'm';
        else if (elem == TypeString)
            out = 's';
        else if (elem == TypeDesc::NONE)
            out = 'x';
        else {
            out = 'x';
            // This only happens in error circumstances. Seems safe to
            // return the code for 'void' and hope everything sorts itself
            // out with the downstream errors.
        }
    }

    if (is_array()) {
        if (is_unsized_array())
            out += "[]";
        else
            out += Strutil::fmt::format("[{}]", arraylength());
    }

    return out;
}



void
TypeSpec::typespecs_from_codes(const char* code, std::vector<TypeSpec>& types)
{
    types.clear();
    while (code && *code) {
        int advance;
        types.push_back(TypeSpec::type_from_code(code, &advance));
        code += advance;
    }
}


bool
equivalent(const StructSpec* a, const StructSpec* b)
{
    OSL_DASSERT(a && b);
    if (a->numfields() != b->numfields())
        return false;
    for (size_t i = 0; i < (size_t)a->numfields(); ++i)
        if (!equivalent(a->field(i).type, b->field(i).type))
            return false;
    return true;
}



bool
equivalent(const TypeSpec& a, const TypeSpec& b)
{
    // The two complex types are equivalent if...
    // they are actually identical (duh)
    if (a == b)
        return true;
    // or if they are structs, and the structs are equivalent
    if (a.is_structure() || b.is_structure()) {
        return a.is_structure() && b.is_structure()
               && a.structspec()->name() == b.structspec()->name()
               && equivalent(a.structspec(), b.structspec());
    }
    // or if the underlying simple types are equivalent
    return ((a.is_vectriple_based() && b.is_vectriple_based())
            || equivalent(a.m_simple, b.m_simple))
           // ... and either both or neither are closures
           && a.is_closure() == b.is_closure()
           // ... and, if arrays, they are the same length, or both unsized,
           //     or one is unsized and the other isn't
           && (a.m_simple.arraylen == b.m_simple.arraylen
               || a.is_unsized_array() != b.is_unsized_array());
}

// Relaxed rules just look to see that the types are isomorphic to each other (ie: same number of base values)
// Note that:
//   * basetypes must match exactly (int vs float vs string)
//   * valuetype cannot be unsized (we must know the concrete number of values)
//   * if paramtype is sized (or not an array) just check for the total number of entries
//   * if paramtype is unsized (shader writer is flexible about how many values come in) -- make sure we are a multiple of the target type
//   * allow a single float setting a vec3 (or equivalent)
bool
relaxed_equivalent(const TypeSpec& a, const TypeSpec& b)
{
    const TypeDesc paramtype = a.simpletype();
    const TypeDesc valuetype = b.simpletype();

    return valuetype.basetype == paramtype.basetype
           && !valuetype.is_unsized_array()
           && ((!paramtype.is_unsized_array()
                && valuetype.basevalues() == paramtype.basevalues())
               || (paramtype.is_unsized_array()
                   && valuetype.basevalues() % paramtype.aggregate == 0)
               || (paramtype.is_vec3() && valuetype == TypeDesc::FLOAT));
}


};  // namespace pvt
OSL_NAMESPACE_EXIT
