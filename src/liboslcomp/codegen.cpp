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

#include <vector>
#include <string>
#include <iostream>

#include <boost/foreach.hpp>

#include "OpenImageIO/dassert.h"
#include "OpenImageIO/strutil.h"

#include "oslcomp_pvt.h"
#include "symtab.h"
#include "ast.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {   // OSL::pvt



size_t
OSLCompilerImpl::add_op_args (size_t nargs, Symbol **args)
{
    size_t n = m_opargs.size ();
    for (size_t i = 0;  i < nargs;  ++i) {
        ASSERT (args[i]);
        m_opargs.push_back (args[i]->dealias());
    }
    return n;
}



void
OSLCompilerImpl::codegen_method (ustring method)
{
    m_codegenmethod = method;
    if (method == main_method_name())
        m_main_method_start = next_op_label ();
}



ustring
OSLCompilerImpl::main_method_name ()
{
    static ustring name ("___main___");
    return name;
}



int
OSLCompilerImpl::emitcode (const char *opname, size_t nargs, Symbol **args,
                           ASTNode *node)
{
    // Reduce to a call to insert_code at the end of the ops
    return insert_code ((int) m_ircode.size(), opname, nargs, args, node);
}



int
OSLCompilerImpl::insert_code (int opnum, const char *opname,
                              size_t nargs, Symbol **args, ASTNode *node)
{
    Opcode op (ustring (opname), m_codegenmethod, m_opargs.size(), nargs);
    if (node)
        op.source (node->sourcefile(), node->sourceline());
    m_ircode.insert (m_ircode.begin()+opnum, op);
    add_op_args (nargs, args);

    // Unless we were inserting at the end, we may need to adjust
    // the jump addresses of other ops and the param init ranges.
    if (opnum < (int)m_ircode.size()-1) {
        // Adjust jump offsets
        for (size_t n = 0;  n < m_ircode.size();  ++n) {
            Opcode &c (m_ircode[n]);
            for (int j = 0; j < (int)Opcode::max_jumps && c.jump(j) >= 0; ++j) {
                if (c.jump(j) > opnum) {
                    c.jump(j) = c.jump(j) + 1;
                    // std::cerr << "Adjusting jump target at op " << n << "\n";
                }
            }
        }
        // Adjust param init ranges
        BOOST_FOREACH (Symbol *s, symtab()) {
            if (s->symtype() == SymTypeParam ||
                  s->symtype() == SymTypeOutputParam) {
                if (s->initbegin() > opnum)
                    s->initbegin (s->initbegin()+1);
                if (s->initend() > opnum)
                    s->initend (s->initend()+1);
            }
        }
    }

    return opnum;
}



Symbol *
OSLCompilerImpl::make_temporary (const TypeSpec &type)
{
    ustring name = ustring::format ("$tmp%d", ++m_next_temp);
    Symbol *s = new Symbol (name, type, SymTypeTemp);
    symtab().insert (s);
    return s;
}



Symbol *
OSLCompilerImpl::make_constant (ustring val)
{
    BOOST_FOREACH (ConstantSymbol *sym, m_const_syms) {
        if (sym->typespec().is_string() && sym->strval() == val)
            return sym;
    }
    // It's not a constant we've added before
    ustring name = ustring::format ("$const%d", ++m_next_const);
    ConstantSymbol *s = new ConstantSymbol (name, val);
    symtab().insert (s);
    m_const_syms.push_back (s);
    return s;
}



Symbol *
OSLCompilerImpl::make_constant (int val)
{
    BOOST_FOREACH (ConstantSymbol *sym, m_const_syms) {
        if (sym->typespec().is_int() && sym->intval() == val)
            return sym;
    }
    // It's not a constant we've added before
    ustring name = ustring::format ("$const%d", ++m_next_const);
    ConstantSymbol *s = new ConstantSymbol (name, val);
    symtab().insert (s);
    m_const_syms.push_back (s);
    return s;
}



Symbol *
OSLCompilerImpl::make_constant (float val)
{
    BOOST_FOREACH (ConstantSymbol *sym, m_const_syms) {
        if (sym->typespec().is_float() && sym->floatval() == val)
            return sym;
    }
    // It's not a constant we've added before
    ustring name = ustring::format ("$const%d", ++m_next_const);
    ConstantSymbol *s = new ConstantSymbol (name, val);
    symtab().insert (s);
    m_const_syms.push_back (s);
    return s;
}



Symbol *
OSLCompilerImpl::make_constant (TypeDesc type, float x, float y, float z)
{
    Vec3 val (x, y, z);
    BOOST_FOREACH (ConstantSymbol *sym, m_const_syms) {
        if (sym->typespec().simpletype() == type && sym->vecval() == val)
            return sym;
    }
    // It's not a constant we've added before
    ustring name = ustring::format ("$const%d", ++m_next_const);
    ConstantSymbol *s = new ConstantSymbol (name, type, x, y, z);
    symtab().insert (s);
    m_const_syms.push_back (s);
    return s;
}



int
ASTNode::emitcode (const char *opname, Symbol *arg0, 
                   Symbol *arg1, Symbol *arg2, Symbol *arg3)
{
    Symbol *args[4] = { arg0, arg1, arg2, arg3 };
    size_t nargs = (arg0 != NULL) + (arg1 != NULL) + 
                   (arg2 != NULL) + (arg3 != NULL);
    return m_compiler->emitcode (opname, nargs, args, this);
}



int
ASTNode::emitcode (const char *opname, size_t nargs, Symbol **args)
{
    return m_compiler->emitcode (opname, nargs, args, this);
}



Symbol *
ASTNode::coerce (Symbol *sym, const TypeSpec &type, bool acceptfloat)
{
    if (equivalent (sym->typespec(), type))
        return sym;   // No coercion necessary

    if (acceptfloat && sym->typespec().is_float())
        return sym;

    if (type.arraylength() == -1 && sym->typespec().is_array() &&
        equivalent (sym->typespec().elementtype(), type.elementtype())) {
        // coercion not necessary to pass known length array to 
        // array parameter of unspecified length.
        return sym;
    }

    if (sym->symtype() == SymTypeConst && sym->typespec().is_int() &&
            type.is_floatbased()) {
        // It's not only the wrong type, it's a constant of the wrong
        // type. We need a new constant of the right type.
        ConstantSymbol *constsym = (ConstantSymbol *) sym;
        sym = m_compiler->make_constant (constsym->floatval ());
        if (type.is_float() || acceptfloat)
            return sym;
    }

    Symbol *t = m_compiler->make_temporary (type);
    emitcode ("assign", t, sym);
    return t;
}



Symbol *
ASTNode::codegen (Symbol *dest)
{
    codegen_children ();
    // FIXME -- nobody should ever call this
    std::cout << "codegen " << nodetypename() << " : " 
              << (opname() ? opname() : "") << "\n";
    return NULL;
}



void
ASTNode::codegen_children ()
{
    BOOST_FOREACH (ref &c, m_children) {
        codegen_list (c);
    }
}



void
ASTNode::codegen_list (ref node)
{
    while (node) {
        node->codegen ();
        node = node->next ();
    }
}



Symbol *
ASTNode::codegen_int (Symbol *, bool boolify, bool invert)
{
    Symbol *dest = codegen ();
    TypeSpec type = dest->typespec ();
    if (! type.is_int() || boolify || invert) {
        // If they're not using an int as the condition, then it's an
        // implied comparison to zero.
        Symbol *tempvar = m_compiler->make_temporary (TypeDesc::TypeInt);
        Symbol *zerovar = NULL;
        if (type.is_string())
            zerovar = m_compiler->make_constant (ustring(""));
        else if (type.is_int())
            zerovar = m_compiler->make_constant ((int)0);
        else
            zerovar = m_compiler->make_constant (0.0f);
        emitcode (invert ? "eq" : "neq", tempvar, dest, zerovar);
        dest = tempvar;
    }
    return dest;
}



Symbol *
ASTshader_declaration::codegen (Symbol *dest)
{
    for (ref f = formals();  f;  f = f->next()) {
        ASSERT (f->nodetype() == ASTNode::variable_declaration_node);
        ASTvariable_declaration *v = (ASTvariable_declaration *) f.get();
        if (v->init()) {
            // If the initializer is a literal and we output it as a
            // constant in the symbol definition, no need for ops.
            std::string out;
            if (v->param_default_literals (v->sym(), out))
                continue;

            if (v->sym()->typespec().is_structure()) {
                // Special case for structs: call codegen_struct_initializers,
                // which will generate init ops for the fields that need them.
                ASTNode::ref finit = v->init();
                if (finit->nodetype() == compound_initializer_node)
                    finit = ((ASTcompound_initializer *)finit.get())->initlist();
                v->codegen_struct_initializers (finit);
                continue;
            }

            m_compiler->codegen_method (v->name());
            v->sym()->initbegin (m_compiler->next_op_label ());
            v->codegen ();
            v->sym()->initend (m_compiler->next_op_label ());
        }
    }

    m_compiler->codegen_method (m_compiler->main_method_name());
    codegen_list (statements());
    return NULL;
}



Symbol *
ASTreturn_statement::codegen (Symbol *dest)
{
    FunctionSymbol *myfunc = oslcompiler->current_function ();
    if (myfunc) {
        // If it's a user function (as opposed to a main shader body)...
        if (expr()) {
            // If we are returning a value, generate code for the value,
            // try to put it in the designated function return location,
            // but if that's not possible, let it go wherever and then
            // copy it.
            ASSERT (myfunc->return_location() != NULL);
            dest = expr()->codegen (myfunc->return_location ());
            if (dest != myfunc->return_location ())
                emitcode ("assign", myfunc->return_location(), dest);
        }
        // Functions that return from their middles are special -- to make
        // them work, we actually wrap them in "dowhile" loops so that we
        // can "break" to exit them early.
        if (myfunc->complex_return ())
            emitcode ("break");
    } else {
        // Must be return from the main shader body -- exit from the shader
        emitcode ("exit");
    }
    return NULL;
}



Symbol *
ASTcompound_initializer::codegen (Symbol *dest)
{
    ASSERT(0 && "compound codegen");
    return NULL;
}



Symbol *
ASTassign_expression::codegen (Symbol *dest)
{
    ASSERT (m_op == Assign);  // all else handled by binary_op

    ASTindex *index = NULL;
    if (var()->nodetype() == index_node) {
        // Assigning to an individual component or array element
        index = (ASTindex *) var().get();
    }
    dest = index ? NULL : var()->codegen();
    Symbol *operand = expr()->codegen (dest);

    if (typespec().is_structure()) {
        // Assignment of struct copies each element individually
        StructSpec *structspec = typespec().structspec ();
        for (int i = 0;  i < (int)structspec->numfields();  ++i) {
            Symbol *dfield, *ofield;
            m_compiler->struct_field_pair (dest, operand, i, dfield, ofield);
            if (dfield->typespec().is_array()) {
                // field is an array
                TypeSpec elemtype = dfield->typespec().elementtype();
                Symbol *tmp = m_compiler->make_temporary (elemtype);
                for (int e = 0;  e < dfield->typespec().arraylength();  ++e) {
                    Symbol *index = m_compiler->make_constant (e);
                    emitcode ("aref", tmp, ofield, index);
                    emitcode ("aassign", dfield, index, tmp);
                }
            } else {
                // field is a scalar
                emitcode ("assign", dfield, ofield);
            }
        }
        return dest;
    }


    if (index)
        index->codegen_assign (operand);
    else if (operand != dest)
        emitcode ("assign", dest, operand);
    return dest;
}



bool
ASTvariable_declaration::param_one_default_literal (const Symbol *sym,
                                                    ASTNode *init,
                                                    std::string &out)
{
    // FIXME -- this only works for single values or arrays made of
    // literals.  Needs to be seriously beefed up.
    ASTliteral *lit = dynamic_cast<ASTliteral *>(init);
    bool completed = true;  // have we output the full initialization?
    TypeSpec type = sym->typespec().elementtype();
    if (type.is_closure()) {
        // this clause avoid trouble and assertions if the following
        // is_int(), i_float(), etc, encounter a closure.
        completed = (lit != NULL);
    } else if (type.is_structure()) {
        // No initializers for struct
        completed = false;
    } else if (type.is_int()) {
        if (lit && lit->typespec().is_int())
            out += Strutil::format ("%d ", lit->intval());
        else {
            out += "0 ";  // FIXME?
            completed = false;
        }
    } else if (type.is_float()) {
        if (lit && lit->typespec().is_int())
            out += Strutil::format ("%d ", lit->intval());
        else if (lit && lit->typespec().is_float())
            out += Strutil::format ("%.8g ", lit->floatval());
        else {
            out += "0 ";  // FIXME?
            completed = false;
        }
    } else if (type.is_triple()) {
        if (lit && lit->typespec().is_int()) {
            float f = lit->intval();
            out += Strutil::format ("%.8g %.8g %.8g ", f, f, f);
        } else if (lit && lit->typespec().is_float()) {
            float f = lit->floatval();
            out += Strutil::format ("%.8g %.8g %.8g ", f, f, f);
        } else if (init && init->typespec() == type &&
                   init->nodetype() == ASTNode::type_constructor_node) {
            ASTtype_constructor *ctr = (ASTtype_constructor *) init;
            ASTNode::ref val = ctr->args();
            float f[3];
            int nargs = 0;
            for (int c = 0;  c < 3;  ++c) {
                if (val.get())
                    ++nargs;
                if (val.get() && val->nodetype() == ASTNode::literal_node) {
                    f[c] = ((ASTliteral *)val.get())->floatval ();
                    val = val->next();
                } else {
                    f[c] = 0;
                    completed = false;
                }
            }
            if (nargs == 1)
                out += Strutil::format ("%.8g %.8g %.8g ", f[0], f[0], f[0]);
            else
                out += Strutil::format ("%.8g %.8g %.8g ", f[0], f[1], f[2]);
        } else {
            out += "0 0 0 ";
            completed = false;
        }
    } else if (type.is_matrix()) {
        float f = 0;
        if (lit && lit->typespec().is_int())
            f = lit->intval();
        else if (lit && lit->typespec().is_float())
            f = lit->floatval();
        else {
            f = 0;  // FIXME?
            completed = false;
        }
        out += Strutil::format ("%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g ",
                                f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f);
    } else if (type.is_string()) {
        if (lit && lit->typespec().is_string())
            out += Strutil::format ("\"%s\" ", lit->strval());
        else {
            out += "\"\" ";  // FIXME?
            completed = false;
        }
    }
    else {
        ASSERT (0 && "help with initializer");
    }
    return completed;
}



bool
ASTvariable_declaration::param_default_literals (const Symbol *sym, std::string &out)
{
    out.clear ();

    // Case 1: Normal vars with initializers, not a struct field --
    // generate them (but handle arrays)
    if (init() && sym->fieldid() < 0) {
        // If it's a compound initializer, look at the individual pieces
        ref init = this->init();
        if (init->nodetype() == compound_initializer_node)
            init = ((ASTcompound_initializer *)init.get())->initlist();
        bool completed = true;  // have we output the full initialization?
        for (ASTNode::ref i = init;  i;  i = i->next())
            completed &= param_one_default_literal (sym, i.get(), out);
        return completed;
    }

    // Case 2: it's a structure field, we need to walk down the init
    // list for the right field initializer (which may itself be compound
    // if that struct element is an array)
    if (init() && sym->fieldid() >= 0 &&
            init()->nodetype() == compound_initializer_node) {
        ref init = ((ASTcompound_initializer *)this->init().get())->initlist();
        for (int field = 0;  init && field < sym->fieldid();  ++field)
            init = init->next();
        if (init) {
            if (init->nodetype() == compound_initializer_node) {
                // The field is itself an array
                init = ((ASTcompound_initializer *)init.get())->initlist();
                bool completed = true;
                for (ASTNode::ref i = init;  i;  i = i->next())
                    completed &= param_one_default_literal (sym, i.get(), out);
                return completed;
            } else {
                // Simple initializer for the field
                return param_one_default_literal (sym, init.get(), out);
            }
        }
    }

    // If there are NO initializers, or if we fell through by not
    // knowing how to handle the cases above, we still need to make a
    // usable default.
    return param_one_default_literal (sym, NULL, out);
}



Symbol *
ASTvariable_declaration::codegen (Symbol *)
{
    if (! init())
        return m_sym;

    // If it's a compound initializer, look at the individual pieces
    ref init = this->init();
    if (init->nodetype() == compound_initializer_node) {
        init = ((ASTcompound_initializer *)init.get())->initlist();
    }

    // Handle structure initialization separately
    if (m_sym->typespec().is_structure())
        return codegen_struct_initializers (init);

    codegen_initlist (init, m_typespec, m_sym);

    return m_sym;
}



void
ASTvariable_declaration::codegen_initlist (ref init, TypeSpec type,
                                           Symbol *sym)
{
    // Loop over a list of initializers (it's just 1 if not an array)...
    for (int i = 0;  init;  init = init->next(), ++i) {
        Symbol *dest = init->codegen (sym);
        if (dest != sym) {
            if (sym->typespec().is_array()) {
                // Array variable -- assign to the i-th element
                TypeSpec elemtype = sym->typespec().elementtype();
                if (! equivalent (elemtype, dest->typespec())) {
                    // We only allow A[ind] = x if the type of x is
                    // equivalent to that of A's elements.  You can't,
                    // for example, do floatarray[ind] = int.  So we 
                    // convert through a temp.
                    Symbol *tmp = dest;
                    dest = m_compiler->make_temporary (elemtype);
                    emitcode ("assign", dest, tmp);
                }
                emitcode ("aassign", sym, m_compiler->make_constant(i), dest);
            } else {
                // Non-array variable, just a simple assignment
                emitcode ("assign", sym, dest);
            }
        }
    }        
}



Symbol *
ASTvariable_declaration::codegen_struct_initializers (ref init)
{
    if (! init->next() && init->typespec() == m_typespec) {
        // Special case: just one initializer, it's a whole struct of
        // the right type.
        Symbol *initsym = init->codegen (m_sym);
        if (initsym != m_sym) {
            StructSpec *structspec (m_typespec.structspec());
            for (int i = 0;  i < (int)structspec->numfields();  ++i) {
                Symbol *symfield, *initfield;
                m_compiler->struct_field_pair (m_sym, initsym, i,
                                               symfield, initfield);
                emitcode ("assign", symfield, initfield);
            }
        }
        return m_sym;
    }

    // General case -- per-field initializers

    bool paraminit = (m_compiler->codegen_method() != m_compiler->main_method_name() &&
                      (m_sym->symtype() == SymTypeParam ||
                       m_sym->symtype() == SymTypeOutputParam));
    for (int i = 0;  init;  init = init->next(), ++i) {
        // Structure element -- assign to the i-th member field
        StructSpec *structspec (m_typespec.structspec());
        const StructSpec::FieldSpec &field (structspec->field(i));
        ustring fieldname = ustring::format ("%s.%s", m_sym->mangled().c_str(),
                                             field.name.c_str());
        Symbol *fieldsym = m_compiler->symtab().find_exact (fieldname);

        if (paraminit) {
            // For parameter initialization, don't really generate ops if it
            // can be statically initialized.
            std::string out;
            if (param_one_default_literal (fieldsym, init.get(), out))
                continue;

            // Delineate and remember the init ops for this field individually
            m_compiler->codegen_method (fieldname);
            fieldsym->initbegin (m_compiler->next_op_label ());
        }

        if (init->nodetype() == compound_initializer_node) {
            // Initialize the field with a compound initializer
            codegen_initlist (((ASTcompound_initializer *)init.get())->initlist(),
                              field.type, fieldsym);
        } else {
            // Initialize the field with a scalar initializer
            Symbol *dest = init->codegen (fieldsym);
            if (dest != fieldsym)
                emitcode ("assign", fieldsym, dest);
        }

        if (paraminit)
            fieldsym->initend (m_compiler->next_op_label ());
    }
    return m_sym;
}



Symbol *
ASTvariable_ref::codegen (Symbol *)
{
    return m_sym;
}



Symbol *
ASTpreincdec::codegen (Symbol *)
{
    Symbol *sym = var()->codegen ();
    Symbol *one = sym->typespec().is_int() ? m_compiler->make_constant(1)
                                           : m_compiler->make_constant(1.0f);
    emitcode (m_op == Incr ? "add" : "sub", sym, sym, one);
    // FIXME -- what if it's an indexed lvalue, like v[i]?
    return sym;
}



Symbol *
ASTpostincdec::codegen (Symbol *dest)
{
    Symbol *sym = var()->codegen ();
    Symbol *one = sym->typespec().is_int() ? m_compiler->make_constant(1)
                                           : m_compiler->make_constant(1.0f);
    if (! dest)
        dest = m_compiler->make_temporary (sym->typespec());
    emitcode ("assign", dest, sym);
    emitcode (m_op == Incr ? "add" : "sub", sym, sym, one);
    // FIXME -- what if it's an indexed lvalue, like v[i]?
    return dest;
}



Symbol *
ASTindex::codegen (Symbol *dest)
{
    // All heavy lifting is done by the version that stores index symbols.
    Symbol *ind = NULL, *ind2 = NULL, *ind3 = NULL;
    return codegen (dest, ind, ind2, ind3);
}



Symbol *
ASTindex::codegen (Symbol *dest, Symbol * &ind,
                   Symbol * &ind2, Symbol *&ind3)
{
    Symbol *lv = lvalue()->codegen ();
    ind = index()->codegen ();
    ind2 = index2() ? index2()->codegen () : NULL;
    ind3 = index3() ? index3()->codegen () : NULL;
    if (! dest)
        dest = m_compiler->make_temporary (typespec());
    if (lv->typespec().is_array()) {
        if (index3()) {
            // matrixarray[a][c][r]
            Symbol *tmp = m_compiler->make_temporary (lv->typespec().elementtype());
            emitcode ("aref", tmp, lv, ind);
            emitcode ("mxcompref", dest, tmp, ind2, ind3);
        } else if (index2()) {
            // colorarray[a][c]
            Symbol *tmp = m_compiler->make_temporary (lv->typespec().elementtype());
            emitcode ("aref", tmp, lv, ind);
            emitcode ("compref", dest, tmp, ind2);
        } else {
            // regulararray[a]
            emitcode ("aref", dest, lv, ind);
        }
    } else if (lv->typespec().is_triple()) {
        emitcode ("compref", dest, lv, ind);
    } else if (lv->typespec().is_matrix()) {
        emitcode ("mxcompref", dest, lv, ind, ind2);
    } else {
        ASSERT (0);
    }
    return dest;
}



void
ASTindex::codegen_assign (Symbol *src, Symbol *ind,
                          Symbol *ind2, Symbol *ind3)
{
    Symbol *lv = lvalue()->codegen ();
    if (! ind)
        ind = index()->codegen ();
    if (! ind2)
        ind2 = index2() ? index2()->codegen () : NULL;
    if (! ind3)
        ind3 = index3() ? index3()->codegen () : NULL;
    if (lv->typespec().is_array()) {
        TypeSpec elemtype = lv->typespec().elementtype();
        if (ind3 && elemtype.is_matrix()) {
            // Component of matrix array, e.g., matrixarray[i][c][r] = float
            Symbol *temp = m_compiler->make_temporary (elemtype);
            emitcode ("aref", temp, lv, ind);
            emitcode ("mxcompassign", temp, ind2, ind3, src);
            emitcode ("aassign", lv, ind, temp);
        } else if (ind2 && elemtype.is_triple()) {
            // Component of triple array, e.g., colorarray[i][c] = float
            Symbol *temp = m_compiler->make_temporary (elemtype);
            emitcode ("aref", temp, lv, ind);
            emitcode ("compassign", temp, ind2, src);
            emitcode ("aassign", lv, ind, temp);
        }
        else if (! equivalent (elemtype, src->typespec())) {
            // Type conversion, e.g., colorarray[i] = float or 
            //    floatarray[i] = int
            // We only allow A[ind] = x if the type of x is equivalent
            // to that of A's elements.  You can't, for example, do
            // floatarray[ind] = int.  So we convert through a temp.
            Symbol *tmp = src;
            src = m_compiler->make_temporary (elemtype);
            emitcode ("assign", src, tmp);
            emitcode ("aassign", lv, ind, src);
        } else {
            // Simple Xarray[i] = X
            emitcode ("aassign", lv, ind, src);
        }
    } else if (lv->typespec().is_triple()) {
        emitcode ("compassign", lv, ind, src);
    } else if (lv->typespec().is_matrix()) {
        emitcode ("mxcompassign", lv, ind, ind2, src);
    } else {
        ASSERT (0);
    }
}



Symbol *
ASTstructselect::codegen (Symbol *dest)
{
    return m_mangledsym;
}



Symbol *
ASTconditional_statement::codegen (Symbol *)
{
    Symbol *condvar = cond()->codegen_int ();

    // Generate the op for the 'if' itself.  Record its label, so that we
    // can go back and patch it with the jump destinations.
    int ifop = emitcode ("if", condvar);
    // "if" is unusual in that it doesn't write its first argument
    oslcompiler->lastop().argread (0, true);
    oslcompiler->lastop().argwrite (0, false);

    // Generate the code for the 'true' and 'false' code blocks, recording
    // the jump destinations for 'else' and the next op after the if.
    oslcompiler->push_nesting (false);
    codegen_list (truestmt());
    int falselabel = m_compiler->next_op_label ();
    codegen_list (falsestmt());
    int donelabel = m_compiler->next_op_label ();
    oslcompiler->pop_nesting (false);

    // Fix up the 'if' to have the jump destinations.
    m_compiler->ircode(ifop).set_jump (falselabel, donelabel);

    // FIXME -- account for the fact that the first argument, unlike
    // almost all other ops, is read, not written
    return NULL;
}



// while (cond) statement
// do statement while (cond);
// for (init; cond; iter);
Symbol *
ASTloop_statement::codegen (Symbol *)
{
    // Generate the op for the loop itself.  Record its label, so that we
    // can go back and patch it with the jump destinations.
    int loop_op = emitcode (opname());
    // Loop ops read their first arg, not write it
    oslcompiler->lastop().argread (0, true);
    oslcompiler->lastop().argwrite (0, false);
        
    oslcompiler->push_nesting (true);
    codegen_list (init());

    int condlabel = m_compiler->next_op_label ();
    Symbol *condvar = cond()->codegen_int ();

    // Retroactively add the argument
    size_t argstart = m_compiler->add_op_args (1, &condvar);
    m_compiler->ircode(loop_op).set_args (argstart, 1);
    // N.B. the arg is both read and written -- already the default state

    int bodylabel = m_compiler->next_op_label ();
    codegen_list (stmt());
    int iterlabel = m_compiler->next_op_label ();
    codegen_list (iter());
    int donelabel = m_compiler->next_op_label ();
    oslcompiler->pop_nesting (true);

    // Fix up the loop op to have the jump destinations.
    m_compiler->ircode(loop_op).set_jump (condlabel, bodylabel,
                                          iterlabel, donelabel);

    // FIXME -- account for the fact that the first argument, unlike
    // almost all other ops, is read, not written
    return NULL;
}



Symbol *
ASTunary_expression::codegen (Symbol *dest)
{
    // Code generation for unary expressions (-x, !x, etc.)

    if (m_op == Not) {
        // Special case for logical ops
        return expr()->codegen_int (NULL, true /*boolify*/, true /*invert*/);
    }

    // Generate the code for our expression
    Symbol *esym = expr()->codegen ();

    if (m_op == Add) {
        // Special case: +x just returns x.
        return esym;
    }

    // If we were not given a requested destination, or if it is not of
    // the right type, make a temporary.
    if (dest == NULL || ! equivalent (dest->typespec(), typespec()))
        dest = m_compiler->make_temporary (typespec());

    // FIXME -- what about coerced types, do we need a temp and copy here?

    // Generate the opcode
    emitcode (opword(), dest, esym);

    return dest;
}



Symbol *
ASTbinary_expression::codegen (Symbol *dest)
{
    // Special case for logal ops that short-circuit
    if (m_op == And || m_op == Or)
        return codegen_logic (dest);

    Symbol *lsym = left()->codegen ();
    Symbol *rsym = right()->codegen ();
    if (dest == NULL || ! equivalent (dest->typespec(), typespec()))
        dest = m_compiler->make_temporary (typespec());

    // FIXME -- what about coerced types, do we need a temp and copy here?

    // Promote ints to float-like types, for mixed arithmetic
    if ((m_op == Mul || m_op == Div || m_op == Add || m_op == Sub)) {
        if ((lsym->typespec().is_closure() || lsym->typespec().is_floatbased()) && rsym->typespec().is_int()) {
            if (rsym->symtype() == SymTypeConst) {
                float val = ((ConstantSymbol *)rsym)->floatval();
                rsym = m_compiler->make_constant (val);
            } else {
                Symbol *tmp = rsym;
                rsym = m_compiler->make_temporary (lsym->typespec());
                emitcode ("assign", rsym, tmp);  // type coercion
            }
        } else if (lsym->typespec().is_int() && (rsym->typespec().is_closure() || rsym->typespec().is_floatbased())) {
            if (lsym->symtype() == SymTypeConst) {
                float val = ((ConstantSymbol *)lsym)->floatval();
                lsym = m_compiler->make_constant (val);
            } else {
                Symbol *tmp = lsym;
                lsym = m_compiler->make_temporary (rsym->typespec());
                emitcode ("assign", lsym, tmp);  // type coercion
            }
        }
    }

    emitcode (opword(), dest, lsym, rsym);
    return dest;
}



Symbol *
ASTbinary_expression::codegen_logic (Symbol *dest)
{
    if (dest == NULL || ! equivalent (dest->typespec(), typespec()))
        dest = m_compiler->make_temporary (typespec());
    Symbol *lsym = left()->codegen_int (dest);

    int ifop = emitcode ("if", lsym);
    // "if" is unusual in that it doesn't write its first argument
    oslcompiler->lastop().argread (0, true);
    oslcompiler->lastop().argwrite (0, false);
    int falselabel;
    m_compiler->push_nesting (false);

    if (m_op == And) {
        Symbol *rsym = right()->codegen_int ();
        // Fixme -- make sure it's an int
        emitcode ("and", dest, lsym, rsym);
        falselabel = m_compiler->next_op_label ();
        emitcode ("assign", dest, m_compiler->make_constant((int)0));
    } else { /* Or */
        emitcode ("assign", dest, m_compiler->make_constant((int)1));
        falselabel = m_compiler->next_op_label ();
        Symbol *rsym = right()->codegen_int ();
        emitcode ("or", dest, rsym, rsym);
    }

    int donelabel = m_compiler->next_op_label ();
    m_compiler->pop_nesting (false);
    m_compiler->ircode(ifop).set_jump (falselabel, donelabel);
    return dest;
}



Symbol *
ASTtypecast_expression::codegen (Symbol *dest)
{
    Symbol *e = expr()->codegen (dest);

    // If the cast is a null operation -- they are already the same types,
    // or we're converting one triple to another -- just pass the expression.
    if (equivalent (typespec(), e->typespec()))
        return e;

    // Some actual conversion is necessary.  Generally, our "assign"
    // op can handle it all easily.
    if (dest == NULL || ! equivalent (dest->typespec(), typespec()))
        dest = m_compiler->make_temporary (typespec());
    emitcode ("assign", dest, e);
    return dest;
}



Symbol *
ASTtype_constructor::codegen (Symbol *dest)
{
    if (dest == NULL || ! equivalent (dest->typespec(), typespec()))
        dest = m_compiler->make_temporary (typespec());

    // Handle simple case of a triple constructed from 3 float literals
    if (typespec().is_triple()) {
        bool all_literals = true;
        ASTNode::ref val = args();
        float f[3];
        for (int c = 0;  c < 3;  ++c) {
            if (val->nodetype() == ASTNode::literal_node &&
                (val->typespec().is_float() || val->typespec().is_int()))
                f[c] = ((ASTliteral *)val.get())->floatval ();
            else
                all_literals = false;
            if (val->next())
                val = val->next();
        }
        if (all_literals)
            return m_compiler->make_constant (typespec().simpletype(),
                                              f[0], f[1], f[2]);
        // Doesn't fit the pattern, drop to the usual case...
    }

    std::vector<Symbol *> argdest;
    argdest.push_back (dest);
    int nargs = 0;
    for (ref a = args();  a;  a = a->next(), ++nargs) {
        Symbol *argval = a->codegen();
        if (argval->typespec().is_int() && !typespec().is_int()) {
            // Coerce to float if it's an int
            if (a->nodetype() == literal_node) {
                // It's a literal int, so let's make a literal float
                int i = ((ASTliteral *)a.get())->intval ();
                argval = m_compiler->make_constant ((float)i);
            } else {
                // General case
                Symbol *tmp = argval;
                argval = m_compiler->make_temporary (TypeSpec(TypeDesc::FLOAT));
                emitcode ("assign", argval, tmp);
            }
        }
        argdest.push_back (argval);
    }
    if (nargs == 1)
        emitcode ("assign", argdest.size(), &argdest[0]);
    else
        emitcode (typespec().string().c_str(), argdest.size(), &argdest[0]);
    return dest;
}



bool
ASTfunction_call::argread (int arg) const
{
    if (is_user_function()) {
        // assume all are readable except return value
        if (! typespec().is_void() && arg == 0)
            return false;
        else
            return true;
    } else {  // built-in function
        return (arg < 32) ? (m_argread & (1 << arg)) : true;
    }
}



bool
ASTfunction_call::argwrite (int arg) const
{
    if (is_user_function()) {
        // assume all are readable except return value
        if (typespec().is_void()) {
            ASTvariable_declaration *formal = (ASTvariable_declaration *)
                list_nth (user_function()->formals(), arg);
            return formal->is_output ();
        } else {
            if (arg == 0)
                return true;  // return value always writes
            ASTvariable_declaration *formal = (ASTvariable_declaration *)
                list_nth (user_function()->formals(), arg-1);
            return formal->is_output ();
        }
    } else {  // built-in function
        return (arg < 32) ? (m_argwrite & (1 << arg)) : false;
    }
}



Symbol *
ASTfunction_call::codegen (Symbol *dest)
{
    // Set up a return destination if not passed one (or not the right type)
    if (! typespec().is_void()) {
        if (dest == NULL || ! equivalent (dest->typespec(), typespec()))
            dest = m_compiler->make_temporary (typespec());
    }

    std::vector<TypeSpec> polyargs;
    m_compiler->typespecs_from_codes (func()->argcodes().c_str()+1, polyargs);

    // Generate code for all the individual arguments.  Remember the
    // individual indices for arguments that are array elements or
    // vector/color/matrix components.
    size_t nargs = listlength (args());
    std::vector<Symbol *> argdest;
    std::vector<Symbol *> index (nargs, 0), index2(nargs, 0), index3(nargs, 0);
    bool indexed_output_params = false;
    int argdest_return_offset = 0;
    ASTNode *a = args().get();

    int returnarg = !typespec().is_void();
    ASTNode *form = is_user_function() ? user_function()->formals().get() : NULL;
    for (int i = 0;  a;  a = a->nextptr(), ++i) {
        Symbol *thisarg = NULL;
        if (a->nodetype() == index_node && argwrite(i+returnarg)) {
            // Special case for individual array elements or vec/col/matrix
            // components being passed as output params of the function --
            // these aren't really lvalues, so we need to restore their
            // values.  We save the indices we genearate code for here...
            ASTindex *indexnode = dynamic_cast<ASTindex *> (a);
            thisarg = indexnode->codegen (NULL, index[i], index2[i], index3[i]);
            indexed_output_params = true;
        } else {
            thisarg = a->codegen ();
        }
        // Handle type coercion of the argument
        if (i < (int)polyargs.size() &&
                polyargs[i].simpletype() != TypeDesc(TypeDesc::UNKNOWN) &&
                polyargs[i].simpletype() != TypeDesc(TypeDesc::UNKNOWN, -1)) {
            Symbol *origarg = thisarg;
            thisarg = coerce (thisarg, polyargs[i]);
            // Error to type-coerce an output -- where would the result go?
            if (thisarg != origarg && form &&
                    ! equivalent (origarg->typespec(), form->typespec()) &&
                    form->nodetype() == variable_declaration_node &&
                    ((ASTvariable_declaration *)form)->is_output()) {
                error ("Cannot pass '%s %s' as argument %d to %s\n\t"
                       "because it is an output parameter that must be a %s",
                       origarg->typespec().c_str(), origarg->name().c_str(),
                       i+1, user_function()->func()->name().c_str(),
                       form->typespec().c_str());
            }
        }
        argdest.push_back (thisarg);
        if (form)
            form = form->nextptr();
    }

    if (is_user_function ()) {
        // Record the return location
        func()->return_location (typespec().is_void() ? NULL : dest);

        // Alias each function formal parameter to the symbol holding
        // the corresponding actual parameter.
        ASTNode *form = user_function()->formals().get();
        ASTNode *a = args().get();
        for (int i = 0;  a;  a = a->nextptr(), form = form->nextptr(), ++i) {
            ASTvariable_declaration *f = (ASTvariable_declaration *) form;
            f->sym()->alias (argdest[i]);

            // If the formal parameter is a struct, we also need to alias
            // each of the fields
            const TypeSpec &ftype (f->sym()->typespec());
            if (ftype.is_structure()) {
                StructSpec *structspec (ftype.structspec());
                for (int fi = 0;  fi < (int)structspec->numfields();  ++fi) {
                    Symbol *fsym, *asym;
                    m_compiler->struct_field_pair (f->sym(), argdest[i], fi,
                                                   fsym, asym);
                    fsym->alias (asym);
                }
            }
        }

        // Return statements inside the middle of a function (not the
        // last statement in the function, or inside a conditional)
        // require special care, since we don't have a general "jump"
        // instruction.  Instead, we wrap the function call inside a
        // do-while loop and "break".
        int loop_op = -1;
        int startlabel = m_compiler->next_op_label ();
        if (func()->complex_return ())
            loop_op = emitcode ("dowhile");

        // Generate the code for the function body
        oslcompiler->push_function (func ());
        codegen_list (user_function()->statements());
        oslcompiler->pop_function ();

        if (func()->complex_return ()) {
            // Second half of the "do-while-break" technique for functions
            // that do not have simple return patterns.  Now we need to
            // retroactively add the loop arguments and jump targets to
            // the loop instruction.
            Symbol *condvar = m_compiler->make_constant (0);
            size_t argstart = m_compiler->add_op_args (1, &condvar);
            m_compiler->ircode(loop_op).set_args (argstart, 1);
            m_compiler->ircode(loop_op).argread (0, true);    // read
            m_compiler->ircode(loop_op).argwrite (0, false);  // not written
            int endlabel = m_compiler->next_op_label ();
            m_compiler->ircode(loop_op).set_jump (startlabel, startlabel,
                                                  endlabel, endlabel);
        }

    } else {
        // Built-in function
        if (! typespec().is_void()) {    // Insert the return dest if non-void
            argdest.insert (argdest.begin(), dest);
            argdest_return_offset = 1;
        }
        // Emit the actual op
        emitcode (m_name.c_str(), argdest.size(), &argdest[0]);
        // Propagate derivative-taking info to the opcode
        m_compiler->lastop().set_argbits (m_argread, m_argwrite,
                                          m_argtakesderivs);
    }

    if (indexed_output_params) {
        // Second half of the element/component-passed-as-output-param
        // issue -- restore the written values to the right spots.
        a = args().get();
        for (int i = 0;  a;  a = a->nextptr(), ++i) {
            if (index[i]) {
                ASTindex *indexnode = dynamic_cast<ASTindex *> (a);
                ASSERT (indexnode);
                indexnode->codegen_assign (argdest[i+argdest_return_offset],
                                           index[i], index2[i], index3[i]);
            }
        }
    }

    return dest;
}



Symbol *
ASTliteral::codegen (Symbol *dest)
{
    TypeSpec t = typespec();
    if (t.is_string())
        return m_compiler->make_constant (ustring(strval()));
    if (t.is_int())
        return m_compiler->make_constant (intval());
    if (t.is_float())
        return m_compiler->make_constant (floatval());
    ASSERT (0 && "Don't know how to generate code for this literal");
    return NULL;
}



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
