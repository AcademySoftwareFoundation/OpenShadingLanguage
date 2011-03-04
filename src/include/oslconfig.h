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

#ifndef OSLCONFIG_H
#define OSLCONFIG_H

/////////////////////////////////////////////////////////////////////////
/// \file
/// Various compile-time defaults are defined here that could, in
/// principle, be redefined if you are using OSL in some particular
/// renderer that wanted things a different way.
/////////////////////////////////////////////////////////////////////////

// Symbol export defines
#include "export.h"

// All the things we need from Imath
#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathColor.h>
#include <OpenEXR/ImathMatrix.h>

// All the things we need from OpenImageIO
#include <OpenImageIO/errorhandler.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/ustring.h>
#ifdef OIIO_NAMESPACE
namespace OIIO = OIIO_NAMESPACE;
#else
namespace OIIO = OpenImageIO;
#endif

#include "oslversion.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

/// @namespace OSL
/// @brief Main namespace enclosing most OSL functionality.
namespace OSL {


/// By default, we operate with single precision float.  Change this
/// definition to make a shading system that fundamentally operates
/// on doubles.
/// FIXME: it's very likely that all sorts of other things will break
/// if you do this, but eventually we should make sure it works.
typedef float Float;

/// By default, use the excellent Imath vector, matrix, and color types
/// from the IlmBase package from: http://www.openexr.com
///
/// It's permissible to override these types with the vector, matrix,
/// and color classes of your choice, provided that (a) your Vec3 and
/// Color3 have the same data layout as a simple Float[3]; (b) your
/// Matrix44 has the same data layout as Float[4][4]; and (c) your
/// classes have most of the obvious constructors and overloaded
/// operators one would expect from a C++ vector/matrix/color class.
typedef Imath::Vec3<Float>     Vec3;
typedef Imath::Matrix44<Float> Matrix44;
typedef Imath::Color3<Float>   Color3;

/// Assume that we are dealing with OpenImageIO's texture system.  It
/// doesn't literally have to be OIIO's... it just needs to have the
/// same API as OIIO's TextureSystem class, it's a purely abstract class
/// anyway.
typedef OIIO::TextureSystem TextureSystem;
typedef OIIO::TextureOpt TextureOpt;

// And some other things we borrow from OIIO...
typedef OIIO::ErrorHandler ErrorHandler;

#ifdef OIIO_NAMESPACE
using OIIO::TypeDesc;
using OIIO::ustring;
using OIIO::ustringHash;
#endif




// These symbols define whether the shading system uses runflags, point
// indices, or spans.  Exactly one should be set to 1!  
// N.B. Spans not fully supported currently.
#define USE_RUNFLAGS   0
#define USE_RUNINDICES 0
#define USE_RUNSPANS   1


// Temporary measure to identify the branch with LLVM support.  Sorry.
#define LLVM_SUPPORT   1

}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSLCONFIG_H */
