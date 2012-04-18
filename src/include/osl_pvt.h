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

#ifndef OSL_PVT_H
#define OSL_PVT_H

#include "oslconfig.h"

#include "OpenImageIO/dassert.h"

#include <boost/tr1/memory.hpp>
using std::tr1::shared_ptr;


OSL_NAMESPACE_ENTER
namespace pvt {

class ASTNode;
class StructSpec;


/// Kinds of shaders
///
enum ShaderType {
    ShadTypeUnknown, ShadTypeGeneric, ShadTypeSurface, 
    ShadTypeDisplacement, ShadTypeVolume, ShadTypeLight,
    ShadTypeLast
};


/// Convert a ShaderType to a human-readable name ("surface", etc.)
///
const char *shadertypename (ShaderType s);

/// Convert a ShaderType to a human-readable name ("surface", etc.)
///
ShaderType shadertype_from_name (const char *name);



/// Uses of shaders
///
enum ShaderUse {
    ShadUseSurface, ShadUseDisplacement,
    ShadUseLast, ShadUseUnknown = ShadUseLast
};


/// Convert a ShaderUse to a human-readable name ("surface", etc.)
///
const char *shaderusename (ShaderUse s);

/// Convert a ShaderUse to a human-readable name ("surface", etc.)
///
ShaderUse shaderuse_from_name (const char *name);



/// Data type for flags that indicate on a point-by-point basis whether
/// we want computations to be performed.
typedef unsigned char Runflag;

/// Pre-defined values for Runflag's.
///
enum RunflagVal { RunflagOff = 0, RunflagOn = 1 };

/// Data type for indices that indicate on which shading points we want
/// computations to be performed.
typedef int RunIndex;





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
        : m_simple(closure ? TypeDesc::PTR : simple), m_structure(0), m_closure(closure)
    { }

    /// Construct a TypeSpec describing a struct or array of structs,
    /// by supplying the struct name, structure id, and array length
    /// (if it's an array of structures).  If structid == 0, search
    /// the existing table for a (globally) matching name and use that
    /// struct if it exists, otherwise add an entry to the struct table.
    TypeSpec (const char *name, int structid, int arraylen=0);

    /// Express the type as a string
    ///
    std::string string () const;

    /// Express the type as a string (char *).  This is safe, the caller
    /// is not responsible for freeing the characters.
    const char *c_str () const;

    /// Stream output
    friend std::ostream& operator<< (std::ostream& o, const TypeSpec& t) {
        return (o << t.string());
    }

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
    const TypeDesc &simpletype () const { return m_simple; }

    /// Is this typespec a closure?  (N.B. if so, you can find out what
    /// kind of closure it is with simpletype()).
    bool is_closure () const { return m_closure && !is_array(); }

    /// Is this typespec an array of closures?
    ///
    bool is_closure_array () const { return m_closure && is_array(); }

    /// Is this typespec based on closures (either a scalar or array of
    /// closures)?
    bool is_closure_based () const { return m_closure; }

    /// Is this typespec a single structure?  Caveat: Returns false if
    /// it's an array of structs.  N.B. You can find out which struct
    /// with structure().
    bool is_structure () const { return m_structure > 0 && !is_array(); }

    /// Is this typespec an array of structures?
    ///
    bool is_structure_array () const { return m_structure > 0 && is_array(); }

    /// Return the structure ID of this typespec, or 0 if it's not a
    /// struct.
    int structure () const { return m_structure; }

    /// Return the structspec for this structure.
    ///
    StructSpec *structspec () const { return structspec(m_structure); }

    /// Find a structure record by id number.
    ///
    static StructSpec *structspec (int id) {
        return id ? struct_list()[id].get() : NULL;
    }

    /// Find a structure index by name, or return 0 if not found.
    /// If 'add' is true, add the struct if not already found.
    static int structure_id (const char *name, bool add=false);

    /// Make room for one new structure and return its index.
    ///
    static int new_struct (StructSpec *n);

    /// Return a reference to the structure list.
    ///
    static std::vector<shared_ptr<StructSpec> > & struct_list ();

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
    TypeSpec elementtype () const { TypeSpec t = *this; t.make_array (0); return t; }

    /// Return the aggregateness of the underlying simple type (SCALAR,
    /// VEC3, or MATRIX44).
    TypeDesc::AGGREGATE aggregate () const { return (TypeDesc::AGGREGATE)m_simple.aggregate; }

    // Note on the is_<simple_type> routines:
    // We don't need to explicitly check for !is_struct(), since the
    // m_simple is always UNKNOWN for structures.

    /// Is it a simple scalar int?
    ///
    bool is_int () const {
//        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return m_simple == TypeDesc::TypeInt && !is_closure();
    }

    /// Is it a simple scalar float?
    ///
    bool is_float () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return m_simple == TypeDesc::TypeFloat && !is_closure();
    }

    /// Is it a color?
    ///
    bool is_color () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return m_simple == TypeDesc::TypeColor && !is_closure();
    }

    /// Is it a point?
    ///
    bool is_point () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return m_simple == TypeDesc::TypePoint && !is_closure();
    }

    /// Is it a vector?
    ///
    bool is_vector () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return m_simple == TypeDesc::TypeVector && !is_closure();
    }

    /// Is it a normal?
    ///
    bool is_normal () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return m_simple == TypeDesc::TypeNormal && !is_closure();
    }

    /// Is it a simple string?
    ///
    bool is_string () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return m_simple == TypeDesc::TypeString && !is_closure();
    }

    /// Is it a string or an array of strings?
    ///
    bool is_string_based () const {
        return m_simple.basetype == TypeDesc::STRING;
    }

    /// Is it a void?
    ///
    bool is_void () const {
        return m_simple == TypeDesc::NONE;
    }

    /// Is it a simple triple (color, point, vector, or normal)?
    ///
    bool is_triple () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return ! is_closure() && 
            (m_simple == TypeDesc::TypeColor ||
             m_simple == TypeDesc::TypePoint ||
             m_simple == TypeDesc::TypeVector ||
             m_simple == TypeDesc::TypeNormal);
    }

    /// Is this a simple type based on floats (including color/vector/etc)?  
    /// This will return false for a closure or array (even if of floats)
    /// or struct.
    bool is_floatbased () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return ! is_closure() && ! is_array() &&
            m_simple.basetype == TypeDesc::FLOAT;
    }

    /// Is it a simple numeric type (based on float or int, even if an
    /// aggregate)?  This is false for a closure or array (even if of
    /// an underlying numeric type) or struct.
    bool is_numeric () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return ! is_closure() && ! is_array() &&
            (m_simple.basetype == TypeDesc::FLOAT || m_simple.basetype == TypeDesc::INT);
    }

    bool is_scalarnum () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return is_numeric() && m_simple.aggregate == TypeDesc::SCALAR;
    }

    /// Is it a simple straight-up single int or float)?
    ///
    bool is_int_or_float () const { return is_scalarnum(); }

    /// Is it a simple vector-like triple (point, vector, or normal, but
    /// not an array or closure)?
    bool is_vectriple () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return ! is_closure() && 
            (m_simple == TypeDesc::TypePoint ||
             m_simple == TypeDesc::TypeVector ||
             m_simple == TypeDesc::TypeNormal);
    }

    /// Is it based on a vector-like triple (point, vector, or normal)?
    /// (It's ok for it to be an array or closure.)
    bool is_vectriple_based () const {
        return (m_simple.elementtype() == TypeDesc::TypePoint ||
                m_simple.elementtype() == TypeDesc::TypeVector ||
                m_simple.elementtype() == TypeDesc::TypeNormal);
    }

    /// Is it a simple matrix (but not an array or closure)?
    ///
    bool is_matrix () const {
        ASSERT (! is_closure() && "Don't call this if it could be a closure");
        return ! is_closure() && 
            m_simple == TypeDesc::TypeMatrix;
    }

    /// Is it a color closure?
    ///
    bool is_color_closure () const {
        return is_closure();
    }

    /// Types are equivalent if they are identical, or if both are
    /// vector-like (and match their array-ness and closure-ness), or
    /// if both are structures with matching fields.
    friend bool equivalent (const TypeSpec &a, const TypeSpec &b);

    /// Is type src is assignable to dst?  It is if they are the equivalent(),
    /// or if dst is a float or float-aggregate and src is a float or int.
    friend bool assignable (const TypeSpec &dst, const TypeSpec &src) {
        if (dst.is_closure() || src.is_closure())
            return (dst.is_closure() && src.is_closure());
        return equivalent (dst, src) || 
            (dst.is_floatbased() && (src.is_float() || src.is_int()));
    }

private:
    TypeDesc m_simple;     ///< Data if it's a simple type
    short m_structure;     ///< 0 is not a structure, >=1 for structure id
    bool  m_closure;       ///< Is it a closure? (m_simple also used)
};



/// Describe the layout of an OSL 'struct'.
/// Basically it's just a list of all the individual fields' names and
/// types.
class StructSpec {
public:
    /// Construct a new struct with the given name, in the given scope.
    ///
    StructSpec (ustring name, int scope) : m_name(name), m_scope(scope) { }

    /// Description of a single structure field -- just a type and name.
    ///
    struct FieldSpec {
        FieldSpec (const TypeSpec &t, ustring n) : type(t), name(n) { }
        TypeSpec type;
        ustring name;
    };

    /// Append a new field (with type and name) to this struct.
    ///
    void add_field (const TypeSpec &type, ustring name) {
        m_fields.push_back (FieldSpec (type, name));
    }

    /// The name of this struct (may not be unique across all scopes).
    ///
    ustring name () const { return m_name; }

    /// The unique mangled name (with scope embedded) of this struct.
    ///
    std::string mangled () const;

    /// The scope number where this struct was defined.
    ///
    int scope () const { return m_scope; }

    /// Number of fields in the struct.
    ///
    size_t numfields () const { return m_fields.size(); }

    /// Return a reference to an individual FieldSpec for one field
    /// of the struct, indexed numerically (starting with 0).
    const FieldSpec & field (size_t i) const { return m_fields[i]; }

private:
    ustring m_name;                    ///< Structure name (unmangled)
    int m_scope;                       ///< Structure's scope id
    std::vector<FieldSpec> m_fields;   ///< List of fields of the struct
};



/// The compiler (or runtime) record of a single symbol (identifier) and
/// all relevant information about it.
class Symbol {
public:
    Symbol (ustring name, const TypeSpec &datatype, SymType symtype,
            ASTNode *declaration_node=NULL) 
        : m_data(NULL), m_step(0), m_size((int)datatype.simpletype().size()),
          m_name(name), m_typespec(datatype), m_symtype(symtype),
          m_has_derivs(false), m_const_initializer(false),
          m_connected_down(false),
          m_initialized(false), m_lockgeom(false),
          m_valuesource(DefaultVal), m_fieldid(-1),
          m_scope(0), m_dataoffset(-1), 
          m_node(declaration_node), m_alias(NULL),
          m_initbegin(0), m_initend(0),
          m_firstread(std::numeric_limits<int>::max()), m_lastread(-1),
          m_firstwrite(std::numeric_limits<int>::max()), m_lastwrite(-1)
    { }
    Symbol () { }  // Default ctr needed for vector resize
    ~Symbol () { }

    const Symbol & operator= (const Symbol &a) {
        // Make absolutely sure that symbol copying goes blazingly fast,
        // since by design we have made this structure hold no unique
        // pointers and have no elements that aren't safe to memcpy, even
        // though the compiler probably can't figure that out.
        if (this != &a) memcpy (this, &a, sizeof(Symbol));
        return *this;
    }

    /// The symbol's (unmangled) name, guaranteed unique only within the
    /// symbol's declaration scope.
    const ustring &name () const { return m_name; }

    /// The symbol's name, mangled to incorporate the scope so it will be
    /// a globally unique name.
    std::string mangled () const;

    /// Data type of this symbol.
    ///
    const TypeSpec &typespec () const { return m_typespec; }

    /// Kind of symbol this is (param, local, etc.)
    ///
    SymType symtype () const { return (SymType) m_symtype; }

    /// Reset the symbol type.  Use with caution!
    ///
    void symtype (SymType newsymtype) { m_symtype = newsymtype; }

    /// Numerical ID of the scope in which this symbol was declared.
    ///
    int scope () const { return m_scope; }

    /// Set the scope of this symbol to s.
    ///
    void scope (int s) { m_scope = s; }

    /// Return teh AST node containing the declaration of this symbol.
    /// Use with care!
    ASTNode *node () const { return m_node; }

    /// Is this symbol a function?
    ///
    bool is_function () const { return m_symtype == SymTypeFunction; }

    /// Is this symbol a structure?
    ///
    bool is_structure () const { return m_symtype == SymTypeType; }

    /// Return a ptr to the symbol that this really refers to, tracing
    /// aliases back all the way until it finds a symbol that isn't an
    /// alias for anything else.
    Symbol *dealias () const {
        Symbol *s = const_cast<Symbol *>(this);
        while (s->m_alias)
            s = s->m_alias;
        return s;
    }

    /// Establish that this symbol is really an alias for another symbol.
    ///
    void alias (Symbol *other) {
        DASSERT (other != this);  // circular alias would be bad
        m_alias = other;
    }

    /// Return a string representation ("param", "global", etc.) of the
    /// SymType s.
    static const char *symtype_shortname (SymType s);

    /// Return a string representation ("param", "global", etc.) of this
    /// symbol.
    const char *symtype_shortname () const {
        return symtype_shortname(symtype());
    }

    /// Return a pointer to the symbol's data.
    ///
    void *data () const { return m_data; }

    /// Return a pointer to the symbol's data at a particular index.
    ///
    void *data (int index) const { return (char *)m_data + index*m_step; }

    /// Specify the location of the symbol's data.
    ///
    void data (void *d) { m_data = d; }

    void dataoffset (int d) { m_dataoffset = d; }
    int dataoffset () const { return m_dataoffset; }

    int step () const { return m_step; }
    void step (int newstep) { m_step = newstep; }

    bool is_uniform () const { return m_step == 0; }
    bool is_varying () const { return m_step != 0; }

    bool has_derivs () const { return m_has_derivs; }
    int deriv_step () const { return m_size; /*m_deriv_step;*/ }
    void has_derivs (bool new_derivs) {
        m_has_derivs = new_derivs;
    }
    int size () const { return m_size; }
    void size (size_t newsize) { m_size = (int)newsize; }

    /// Return the size for each point, including derivs.
    ///
    int derivsize () const { return m_has_derivs ? 3*m_size : m_size; }

    bool connected () const { return valuesource() == ConnectedVal; }
    bool connected_down () const { return m_connected_down; }
    void connected_down (bool c) { m_connected_down = c; }

    /// Where did the symbol's value come from?
    ///
    enum ValueSource { DefaultVal, InstanceVal, GeomVal, ConnectedVal };

    ValueSource valuesource () const { return (ValueSource) m_valuesource; }
    void valuesource (ValueSource v) { m_valuesource = v; }
    const char *valuesourcename () const;

    int fieldid () const { return m_fieldid; }
    void fieldid (int id) { m_fieldid = id; }

    int initbegin () const { return m_initbegin; }
    void initbegin (int i) { m_initbegin = i; }
    int initend () const { return m_initend; }
    void initend (int i) { m_initend = i; }
    void set_initrange (int b=0, int e=0) { m_initbegin = b; m_initend = e; }
    bool has_init_ops () const { return m_initbegin != m_initend; }

    /// Clear read/write usage info.
    ///
    void clear_rw () {
        m_firstread = m_firstwrite = std::numeric_limits<int>::max();
        m_lastread = m_lastwrite = -1;
    }
    /// Mark whether the symbol was read and/or written on the given op.
    ///
    void mark_rw (int op, bool read, bool write) {
        if (read) {
            m_firstread = std::min (m_firstread, op);
            m_lastread  = std::max (m_lastread, op);
        }
        if (write) {
            m_firstwrite = std::min (m_firstwrite, op);
            m_lastwrite  = std::max (m_lastwrite, op);
        }
    }

    void union_rw (int fr, int lr, int fw, int lw) {
        m_firstread  = std::min (m_firstread, fr);
        m_lastread   = std::max (m_lastread, lr);
        m_firstwrite = std::min (m_firstwrite, fw);
        m_lastwrite  = std::max (m_lastwrite, lw);
    }

    int firstread () const  { return m_firstread; }
    int lastread () const   { return m_lastread; }
    int firstwrite () const { return m_firstwrite; }
    int lastwrite () const  { return m_lastwrite; }
    int firstuse () const   { return std::min (firstread(), firstwrite()); }
    int lastuse () const    { return std::max (lastread(), lastwrite()); }
    bool everread () const  { return lastread() >= 0; }
    bool everwritten () const { return lastwrite() >= 0; }
    bool everused () const  { return everread() || everwritten(); }
    void set_read (int first, int last) {
        m_firstread = first;  m_lastread = last;
    }
    void set_write (int first, int last) {
        m_firstwrite = first;  m_lastwrite = last;
    }

    bool initialized () const { return m_initialized; }
    void initialized (bool init) { m_initialized = init; }

    bool lockgeom () const { return m_lockgeom; }
    void lockgeom (bool lock) { m_lockgeom = lock; }

    bool is_constant () const { return symtype() == SymTypeConst; }

    /// Stream output
    std::ostream& print (std::ostream& out) const;
    std::ostream& print_vals (std::ostream& out) const;

protected:
    void *m_data;               ///< Pointer to the data
    int m_step;                 ///< Step (in bytes) from point to point
    int m_size;                 ///< Size of data (in bytes)
    ustring m_name;             ///< Symbol name (unmangled)
    TypeSpec m_typespec;        ///< Data type of the symbol
    char m_symtype;             ///< Kind of symbol (param, local, etc.)
    unsigned m_has_derivs:1;    ///< Step to derivs (0 == has no derivs)
    unsigned m_const_initializer:1; ///< initializer is a constant expression
    unsigned m_connected_down:1;///< Connected to a later/downtream layer
    unsigned m_initialized:1;   ///< If a param, has it been initialized?
    unsigned m_lockgeom:1;      ///< Is the param not overridden by geom?
    char m_valuesource;         ///< Where did the value come from?
    short m_fieldid;            ///< Struct field of this var (or -1)
    int m_scope;                ///< Scope where this symbol was declared
    int m_dataoffset;           ///< Offset of the data (-1 for unknown)
    ASTNode *m_node;            ///< Ptr to the declaration of this symbol
    Symbol *m_alias;            ///< Another symbol that this is an alias for
    int m_initbegin, m_initend; ///< Range of init ops (for params)
    int m_firstread, m_lastread;///< First and last op the sym is read
    int m_firstwrite, m_lastwrite;///< First and last op the sym is written
};



typedef std::vector<Symbol> SymbolVec;
typedef Symbol * SymbolPtr;
typedef std::vector<Symbol *> SymbolPtrVec;



/// Intermediate Represenatation opcode
///
class Opcode {
public:
    Opcode (ustring op, ustring method, size_t firstarg=0, size_t nargs=0)
        : m_firstarg((int)firstarg), m_method(method), m_sourceline(0)
    {
        reset (op, nargs);   // does most of the heavy lifting
    }

    void reset (ustring opname, size_t nargs) {
        m_op = opname;
        m_nargs = (int) nargs;
        set_jump ();
        m_argread = ~1; // Default - all args are read except the first
        m_argwrite = 1; // Default - first arg only is written by the op
        m_argtakesderivs = 0; // Default - doesn't take derivs
    }

    ustring opname () const { return m_op; }
    int firstarg () const { return m_firstarg; }
    int nargs () const { return m_nargs; }
    ustring method () const { return m_method; }
    void method (ustring method) { m_method = method; }
    void source (ustring sourcefile, int sourceline) {
        m_sourcefile = sourcefile;
        m_sourceline = sourceline;
    }
    ustring sourcefile () const { return m_sourcefile; }
    int sourceline () const { return m_sourceline; }

    void set_args (size_t firstarg, size_t nargs) {
        m_firstarg = (int) firstarg;
        m_nargs = (int) nargs;
    }

    /// Set the jump addresses (-1 means no jump)
    ///
    void set_jump (int jump0=-1, int jump1=-1, int jump2=-1, int jump3=-1) {
        m_jump[0] = jump0;
        m_jump[1] = jump1;
        m_jump[2] = jump2;
        m_jump[3] = jump3;
    }

    void add_jump (int target) {
        for (unsigned int i = 0;  i < max_jumps;  ++i)
            if (m_jump[i] < 0) {
                m_jump[i] = target;
                return;
            }
    }

    /// Return the i'th jump target address (-1 for none).
    ///
    int jump (int i) const { return m_jump[i]; }
    int &jump (int i) { return m_jump[i]; }

    /// Maximum jump targets an op can have.
    ///
    static const unsigned int max_jumps = 4;

    /// What's the farthest address that we jump to?
    ///
    int farthest_jump () const {
        int f = jump(0);
        for (unsigned int i = 1;  i < max_jumps;  ++i)
            f = std::max (f, jump(i));
        return f;
    }

    /// Is the argument number 'arg' read by the op?
    ///
    bool argread (int arg) const {
        return (arg < 32) ? (m_argread & (1 << arg)) : true;
    }
    /// Is the argument number 'arg' written by the op?
    ///
    bool argwrite (int arg) const {
        return (arg < 32) ? (m_argwrite & (1 << arg)) : false;
    }
    /// Declare that argument number 'arg' is read by this op.
    ///
    void argread (int arg, bool val) {
        if (arg < 32) {
            if (val)
                m_argread |= (1 << arg);
            else
                m_argread &= ~(1 << arg);
        }
    }
    /// Declare that argument number 'arg' is written by this op.
    ///
    void argwrite (int arg, bool val) {
        if (arg < 32) {
            if (val)
                m_argwrite |= (1 << arg);
            else
                m_argwrite &= ~(1 << arg);
        }
    }
    /// Declare that argument number 'arg' is only written (not read!) by
    /// this op.
    void argwriteonly (int arg) {
        argread (arg, false);
        argwrite (arg, true);
    }

    /// Does the argument number 'arg' take derivatives?
    ///
    bool argtakesderivs (int arg) const {
        return (arg < 32) ? (m_argtakesderivs & (1 << arg)) : false;
    }

    /// Declare that argument number 'arg' takes derivatives.
    ///
    void argtakesderivs (int arg, bool val) {
        if (arg < 32) {
            if (val)
                m_argtakesderivs |= (1 << arg);
            else
                m_argtakesderivs &= ~(1 << arg);
        }
    }

    /// Set the read, write, and takesderivs bit fields all at once.
    ///
    void set_argbits (unsigned int read, unsigned int wr, unsigned int deriv) {
        m_argread = read;
        m_argwrite = wr;
        m_argtakesderivs = deriv;
    }

    unsigned int argread_bits () const { return m_argread; }
    unsigned int argwrite_bits () const { return m_argwrite; }

    /// Return the entire argtakesderivs at once with a full bitfield.
    ///
    unsigned int argtakesderivs_all () const { return m_argtakesderivs; }

private:
    ustring m_op;                   ///< Name of opcode
    int m_firstarg;                 ///< Index of first argument
    int m_nargs;                    ///< Total number of arguments
    ustring m_method;               ///< Which param or method this code is for
    int m_jump[max_jumps];          ///< Jump addresses (-1 means none)
    ustring m_sourcefile;           ///< Source filename for this op
    int m_sourceline;               ///< Line of source code for this op
    unsigned int m_argread;         ///< Bit field - which args are read
    unsigned int m_argwrite;        ///< Bit field - which args are written
    unsigned int m_argtakesderivs;  ///< Bit field - which args take derivs
    // N.B. We only have 32 bits for m_argread and m_argwrite.  We live
    // with this, and it's ok because there are very few ops that allow
    // more than 32 args, and those that do are read-only that far out.
    // Seems silly to add complexity here to deal with arbitrary param
    // counts and read/write-ability for cases that never come up.
};


typedef std::vector<Opcode> OpcodeVec;



}; // namespace OSL::pvt
OSL_NAMESPACE_EXIT

#endif /* OSL_PVT_H */
