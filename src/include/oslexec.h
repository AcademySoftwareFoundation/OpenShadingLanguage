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

#ifndef OSLEXEC_H
#define OSLEXEC_H


#include "OpenImageIO/typedesc.h"
#include "OpenImageIO/refcnt.h"



namespace OSL {

namespace pvt {
class ShaderInstance;
typedef shared_ptr<ShaderInstance> ShaderInstanceRef;
};
using pvt::ShaderInstanceRef;



class ShadingSystem
{
public:
    ShadingSystem ();
    virtual ~ShadingSystem ();

    static ShadingSystem *create ();
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
    virtual void Parameter (const char *name, TypeDesc t, const void *val) {}
#if 0
    virtual void Parameter (const char *name, int val) {
        Parameter (name, TypeDesc::IntType, &val);
    }
    virtual void Parameter (const char *name, float val) {
        Parameter (name, TypeDesc::FloatType, &val);
    }
    virtual void Parameter (const char *name, double val) {}
    virtual void Parameter (const char *name, const char *val) {}
    virtual void Parameter (const char *name, const std::string &val) {}
    virtual void Parameter (const char *name, TypeDesc t, const int *val) {}
    virtual void Parameter (const char *name, TypeDesc t, const float *val) {}
    virtual void Parameter (const char *name, TypeDesc t, const char **val) {}
#endif

    /// 
    virtual ShaderInstanceRef Shader (const char *shaderusage,
                                      const char *shadername=NULL,
                                      const char *layername=NULL) = 0;
    virtual void ShaderGroupBegin (void) = 0;
    virtual ShaderInstanceRef ShaderGroupEnd (void) = 0;
    virtual void ConnectShaders (const char *srclayer, const char *srcparam,
                                 const char *dstlayer, const char *dstparam)=0;

    /// If any of the API routines returned false indicating an error,
    /// this routine will return the error string (and clear any error
    /// flags).  If no error has occurred since the last time geterror()
    /// was called, it will return an empty string.
    virtual std::string geterror () const = 0;

    /// Return the statistics output as a huge string.
    ///
    virtual std::string getstats (int level=1) const = 0;

private:
    // Make delete private and unimplemented in order to prevent apps
    // from calling it.  Instead, they should call ShadingSystem::destroy().
    void operator delete (void *todel) { }
};



}; // namespace OSL


#endif /* OSLEXEC_H */
