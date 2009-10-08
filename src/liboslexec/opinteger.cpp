/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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


/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of bitwise integer operations
/// such as &, |, <<, etc.
///
/////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


namespace {

// Functors
class BitAnd {
public:
    BitAnd (ShadingExecution *) { }
    inline void operator() (int &result, int a, int b) { result = a & b; }
};

class BitOr {
public:
    BitOr (ShadingExecution *) { }
    inline void operator() (int &result, int a, int b) { result = a | b; }
};

class Xor {
public:
    Xor (ShadingExecution *) { }
    inline void operator() (int &result, int a, int b) { result = a ^ b; }
};

class Shl {
public:
    Shl (ShadingExecution *) { }
    inline void operator() (int &result, int a, int b) { result = a << b; }
};

class Shr {
public:
    Shr (ShadingExecution *) { }
    inline void operator() (int &result, int a, int b) { result = a >> b; }
};

class Compl {
public:
    Compl (ShadingExecution *) { }
    inline void operator() (int &result, int a) { result = ~a; }
};


};  // End anonymous namespace




DECLOP (OP_bitand)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());
    ASSERT (Result.typespec().is_int() && A.typespec().is_int() &&
            B.typespec().is_int());

    binary_op_guts<int,int,int,BitAnd> (Result, A, B, exec,
                                        runflags, beginpoint, endpoint);
}



DECLOP (OP_bitor)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());
    ASSERT (Result.typespec().is_int() && A.typespec().is_int() &&
            B.typespec().is_int());

    binary_op_guts<int,int,int,BitOr> (Result, A, B, exec,
                                       runflags, beginpoint, endpoint);
}



DECLOP (OP_xor)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());
    ASSERT (Result.typespec().is_int() && A.typespec().is_int() &&
            B.typespec().is_int());

    binary_op_guts<int,int,int,Xor> (Result, A, B, exec,
                                     runflags, beginpoint, endpoint);
}



DECLOP (OP_shl)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());
    ASSERT (Result.typespec().is_int() && A.typespec().is_int() &&
            B.typespec().is_int());

    binary_op_guts<int,int,int,Shl> (Result, A, B, exec,
                                     runflags, beginpoint, endpoint);
}



DECLOP (OP_shr)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
    Symbol &B (exec->sym (args[2]));
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());
    ASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    ASSERT (! B.typespec().is_closure() &&
            ! B.typespec().is_structure() &&
            ! B.typespec().is_array());
    ASSERT (Result.typespec().is_int() && A.typespec().is_int() &&
            B.typespec().is_int());

    binary_op_guts<int,int,int,Shr> (Result, A, B, exec,
                                     runflags, beginpoint, endpoint);
}



DECLOP (OP_compl)
{
    DASSERT (nargs == 2);
#ifdef DEBUG
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));
#endif
    DASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());
    DASSERT (! A.typespec().is_closure() &&
            ! A.typespec().is_structure() &&
            ! A.typespec().is_array());
    DASSERT (Result.typespec().is_int() && A.typespec().is_int());

    unary_op_noderivs<int,int,Compl> (exec, nargs, args,
                                      runflags, beginpoint, endpoint);
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
