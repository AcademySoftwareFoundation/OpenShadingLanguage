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

#ifndef OSLEXEC_H
#define OSLEXEC_H


#include "oslconfig.h"
// osl_pvt.h is required for 'Runflags' definition
#include "osl_pvt.h"

#include "OpenImageIO/refcnt.h"             // just to get shared_ptr from boost ?!
#include "OpenImageIO/ustring.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

class RendererServices;
class ShadingAttribState;
typedef shared_ptr<ShadingAttribState> ShadingAttribStateRef;
class ShaderGlobals;
class ClosureColor;
class ClosureParam;
class OSLCompiler;



/// Opaque pointer to whatever the renderer uses to represent a
/// (potentially motion-blurred) coordinate transformation.
typedef const void * TransformationPtr;


// Callbacks for closure creation
typedef void (*PrepareClosureFunc)(RendererServices *, int id, void *data);
typedef void (*SetupClosureFunc)(RendererServices *, int id, void *data);
typedef bool (*CompareClosureFunc)(int id, const void *dataA, const void *dataB);


class ShadingSystem
{
protected:
    /// ShadingSystem is an abstract class, its constructor is protected
    /// so that ordinary users can't make an instance, but instead are
    /// forced to request one via ShadingSystem::create().
    ShadingSystem ();
    virtual ~ShadingSystem ();

public:
    static ShadingSystem *create (RendererServices *renderer=NULL,
                                  TextureSystem *texturesystem=NULL,
                                  ErrorHandler *err=NULL);
    static void destroy (ShadingSystem *x);

    /// Set an attribute controlling the texture system.  Return true
    /// if the name and type were recognized and the attrib was set.
    /// Documented attributes:
    ///
    virtual bool attribute (const std::string &name, TypeDesc type,
                            const void *val) = 0;
    // Shortcuts for common types
    bool attribute (const std::string &name, int val) {
        return attribute (name, TypeDesc::INT, &val);
    }
    bool attribute (const std::string &name, float val) {
        return attribute (name, TypeDesc::FLOAT, &val);
    }
    bool attribute (const std::string &name, double val) {
        float f = (float) val;
        return attribute (name, TypeDesc::FLOAT, &f);
    }
    bool attribute (const std::string &name, const char *val) {
        return attribute (name, TypeDesc::STRING, &val);
    }
    bool attribute (const std::string &name, const std::string &val) {
        const char *s = val.c_str();
        return attribute (name, TypeDesc::STRING, &s);
    }

    /// Get the named attribute, store it in value.
    ///
    virtual bool getattribute (const std::string &name, TypeDesc type,
                               void *val) = 0;
    // Shortcuts for common types
    bool getattribute (const std::string &name, int &val) {
        return getattribute (name, TypeDesc::INT, &val);
    }
    bool getattribute (const std::string &name, float &val) {
        return getattribute (name, TypeDesc::FLOAT, &val);
    }
    bool getattribute (const std::string &name, double &val) {
        float f;
        bool ok = getattribute (name, TypeDesc::FLOAT, &f);
        if (ok)
            val = f;
        return ok;
    }
    bool getattribute (const std::string &name, char **val) {
        return getattribute (name, TypeDesc::STRING, val);
    }
    bool getattribute (const std::string &name, std::string &val) {
        const char *s = NULL;
        bool ok = getattribute (name, TypeDesc::STRING, &s);
        if (ok)
            val = s;
        return ok;
    }

    /// Set a parameter of the next shader.
    ///
    virtual bool Parameter (const char *name, TypeDesc t, const void *val)
        { return true; }
#if 0
    virtual bool Parameter (const char *name, int val) {
        Parameter (name, TypeDesc::IntType, &val);
    }
    virtual bool Parameter (const char *name, float val) {
        Parameter (name, TypeDesc::FloatType, &val);
    }
    virtual bool Parameter (const char *name, double val) {}
    virtual bool Parameter (const char *name, const char *val) {}
    virtual bool Parameter (const char *name, const std::string &val) {}
    virtual bool Parameter (const char *name, TypeDesc t, const int *val) {}
    virtual bool Parameter (const char *name, TypeDesc t, const float *val) {}
    virtual bool Parameter (const char *name, TypeDesc t, const char **val) {}
#endif

    /// Create a new shader instance, either replacing the one for the
    /// specified usage (if not within a group) or appending to the 
    /// current group (if a group has been started).
    virtual bool Shader (const char *shaderusage,
                         const char *shadername=NULL,
                         const char *layername=NULL) = 0;

    /// Signal the start of a new shader group.
    ///
    virtual bool ShaderGroupBegin (void) = 0;

    /// Signal the end of a new shader group.
    ///
    virtual bool ShaderGroupEnd (void) = 0;

    /// Connect two shaders within the current group
    ///
    virtual bool ConnectShaders (const char *srclayer, const char *srcparam,
                                 const char *dstlayer, const char *dstparam)=0;

    /// Return a reference-counted (but opaque) reference to the current
    /// shading attribute state maintained by the ShadingSystem.
    virtual ShadingAttribStateRef state () const = 0;

    /// Clear the current shading attribute state, i.e., no shaders
    /// specified.
    virtual void clear_state () = 0;

    /// Return the statistics output as a huge string.
    ///
    virtual std::string getstats (int level=1) const = 0;

    virtual void register_closure(const char *name, int id, const ClosureParam *params, int size,
                                  PrepareClosureFunc prepare, SetupClosureFunc setup, CompareClosureFunc compare,
                                  int labels_offset, int max_labels) = 0;

    static void register_builtin_closures(ShadingSystem *ss);
    static void register_builtin_closures(OSLCompiler *cc);

private:
    // Make delete private and unimplemented in order to prevent apps
    // from calling it.  Instead, they should call ShadingSystem::destroy().
    void operator delete (void *todel) { }
};



/// ShaderGlobals holds references to data to be used for the "global"
/// data for each point being shaded -- position, normal, uv
/// coordinates, etc.
class ShaderGlobals
{
public:
    ShaderGlobals () 
        : iscameraray(true), isshadowray(false), flipHandedness(false), backfacing(false)
    { }
    ~ShaderGlobals () { }

    VaryingRef<Vec3> P;                ///< Position
    VaryingRef<Vec3> dPdx, dPdy;       ///< Partials
    VaryingRef<Vec3> I;                ///< Incident ray
    VaryingRef<Vec3> dIdx, dIdy;       ///< Partial derivatives for I
    VaryingRef<Vec3> N;                ///< Shading normal
    VaryingRef<Vec3> Ng;               ///< True geometric normal
    VaryingRef<float> u, v;            ///< Surface parameters
    VaryingRef<float> dudx, dudy;      ///< u differentials
    VaryingRef<float> dvdx, dvdy;      ///< v differentials
    VaryingRef<Vec3> dPdu, dPdv;       ///< Partial derivatives
    VaryingRef<float> time;            ///< Time for each sample
    VaryingRef<float> dtime;           ///< Time interval for each sample
    VaryingRef<Vec3> dPdtime;          ///< Velocity
    VaryingRef<Vec3> Ps;               ///< Point being lit
    VaryingRef<Vec3> dPsdx, dPsdy;     ///< Derivs of Ps

    VaryingRef<void *> renderstate;    ///< Renderer context for each sample

    VaryingRef<TransformationPtr> object2common; /// Object->common xform
    VaryingRef<TransformationPtr> shader2common; /// Shader->common xform

    VaryingRef<ClosureColor *> Ci;     ///< Output colors

    VaryingRef<float> surfacearea;     ///< Total area of the object (not exposed)
    bool iscameraray;                  ///< True if computing for camera ray
    bool isshadowray;                  ///< True if computing for shadow opacity
    bool flipHandedness;               ///< flips the result of calculatenormal()
    bool backfacing;                   ///< True if we want to shade the back face
};



/// RendererServices defines an abstract interface through which a 
/// renderer may provide callback to the ShadingSystem.
class RendererServices {
public:
    RendererServices () { }
    virtual ~RendererServices () { }

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.
    virtual bool get_matrix (Matrix44 &result, TransformationPtr xform,
                             float time) = 0;

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  The default implementation is
    /// to use get_matrix and invert it, but a particular renderer may
    /// have a better technique and overload the implementation.
    virtual bool get_inverse_matrix (Matrix44 &result, TransformationPtr xform,
                                     float time);

    /// Get the 4x4 matrix that transforms points from the named
    /// 'from' coordinate system to "common" space at the given time.
    virtual bool get_matrix (Matrix44 &result, ustring from, float time) = 0;

    /// Get the named attribute from the renderer and if found then
    /// write it into 'val'.  Otherwise, return false.  If no object is
    /// specified (object == ustring()), then the renderer should search *first*
    /// for the attribute on the currently shaded object, and next, if
    /// unsuccessful, on the currently shaded "scene". 
    virtual bool get_attribute (void *renderstate, bool derivatives, 
                                ustring object, TypeDesc type, ustring name, 
                                void *val ) = 0;

    /// Similar to get_attribute();  this method will return the 'index'
    /// element of an attribute array.
    virtual bool get_array_attribute (void *renderstate, bool derivatives, 
                                      ustring object, TypeDesc type, 
                                      ustring name, int index, void *val ) = 0;

    /// Get the named user-data from the current object and write it into
    /// 'val'.  This function provides and array of renderstates 'npoints'
    /// in size, and a block of memory in which to store the user data (pointed
    /// to by val).  Both arrays have a stride of 'stepsize'.
    virtual bool get_userdata (OSL::pvt::Runflag *runflags, int npoints, bool derivatives, 
                               ustring name, TypeDesc type, 
                               void *renderstates,  int renderstates_stepsize,
                               void *val, int val_stepsize ) = 0;

    /// Does the current object have the named user-data associated with it?
    virtual bool has_userdata (ustring name, TypeDesc type, void *renderstate) = 0;

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system to at the given time.  The
    /// default implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual bool get_inverse_matrix (Matrix44 &result, ustring to, float time);

};


}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSLEXEC_H */
