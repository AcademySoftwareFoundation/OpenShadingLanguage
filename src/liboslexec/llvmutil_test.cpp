/*
Copyright (c) 2009-2013 Sony Pictures Imageworks Inc., et al.
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

#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/ustring.h>
#include <OpenImageIO/dassert.h>
#include "OSL/llvm_util.h"



// This demonstrates the use of LLVM_Util to generate the following
// function on the fly, JIT it, and call it:
//      int myadd (int arg1, int arg2)
//      {
//          return arg1 + arg2;    
//      }
//



int
main (int argc, char *argv[])
{
    // Setup
    OSL::pvt::LLVM_Util ll;

    // Make a function with prototype:   int myadd (int arg1, int arg2)
    // and make it the current function.
    llvm::Function *func = ll.make_function ("myadd",         // name
                                             false,           // fastcall
                                             ll.type_int(),   // return
                                             ll.type_int(),   // arg1
                                             ll.type_int());  // arg2
    ll.current_function (func);

    // Generate the ops for this function:  return arg1 + arg2
    llvm::Value *arg1 = ll.current_function_arg (0);
    llvm::Value *arg2 = ll.current_function_arg (1);
    llvm::Value *sum = ll.op_add (arg1, arg2);
    ll.op_return (sum);

    // Optimize it
    ll.setup_optimization_passes (0);
    ll.do_optimize ();

    // Print the optimized bitcode
    std::cout << "Generated the following bitcode:\n"
              << ll.bitcode_string(func) << "\n";

    // Ask for a callable function (will JIT on demand)
    typedef int (*IntFuncOfTwoInts)(int,int);
    IntFuncOfTwoInts myadd = (IntFuncOfTwoInts) ll.getPointerToFunction (func);

    // Call it:
    int result = myadd (13, 29);
    std::cout << "The result is " << result << "\n";
    ASSERT (result == 42);

    return 0;
}
