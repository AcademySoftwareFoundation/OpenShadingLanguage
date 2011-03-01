/*
Copyright 2008-2010 Sony Pictures Imageworks Inc., et al.
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


#ifndef OSL_EXPORT_H
#define OSL_EXPORT_H

/// \file
/// OSLPUBLIC and OSLEXPORT macros that are necessary for proper symbol
/// export when doing multi-platform development.


///
/// On Windows, when compiling code that will end up in a DLL, symbols
/// must be marked as 'exported' (i.e. __declspec(dllexport)) or they
/// won't be visible to programs linking against the DLL.
///
/// In addition, when compiling the application code that calls the DLL,
/// if a routine is marked as 'imported' (i.e. __declspec(dllimport)),
/// the compiler can be smart about eliminating a level of calling
/// indirection.  But you DON'T want to use __declspec(dllimport) when
/// calling a function from within its own DLL (it will still compile
/// correctly, just not with maximal efficiency).  Which is quite the
/// dilemma since the same header file is used by both the library and
/// its clients.  Sheesh!
///
/// We solve this awful mess by defining these macros:
///
/// OSLPUBLIC - normally, assumes that it's being seen by a client
///             of the library, and therefore declare as 'imported'.
///             But if OSL_EXPORT_PUBLIC is defined, change the declaration
///             to 'exported' -- you want to define this macro when
///             compiling the module that actually defines the class.
///
/// Note that on Linux, all symbols are exported so this just isn't a
/// problem, so we define these macros to be nothing.
///
/// It's a shame that we have to clutter all our header files with these
/// stupid macros just because the Windows world is such a mess.
///
/// There is a separate define for each library, because there inter-
/// dependencies, and so what is exported for one may be imported for
/// another.

#ifndef OSLCOMPPUBLIC
#  if defined(_MSC_VER) && defined(_WIN32)
#    if defined(oslcomp_EXPORTS)
#      define OSLCOMPPUBLIC __declspec(dllexport)
#    else
#      define OSLCOMPPUBLIC __declspec(dllimport)
#    endif
#  else
#    define OSLCOMPPUBLIC
#  endif
#endif

#ifndef OSLEXECPUBLIC
#  if defined(_MSC_VER) && defined(_WIN32)
#    if defined(oslexec_EXPORTS)
#      define OSLEXECPUBLIC __declspec(dllexport)
#    else
#      define OSLEXECPUBLIC __declspec(dllimport)
#    endif
#  else
#    define OSLEXECPUBLIC
#  endif
#endif

#ifndef OSLQUERYPUBLIC
#  if defined(_MSC_VER) && defined(_WIN32)
#    if defined(oslquery_EXPORTS)
#      define OSLQUERYPUBLIC __declspec(dllexport)
#    else
#      define OSLQUERYPUBLIC __declspec(dllimport)
#    endif
#  else
#    define OSLQUERYPUBLIC
#  endif
#endif

#endif // OSL_EXPORT_H
