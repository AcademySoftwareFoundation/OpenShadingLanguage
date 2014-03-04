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

#pragma once

#include <map>

#include "OSL/oslexec.h"

OSL_NAMESPACE_ENTER

class SimpleRenderer : public RendererServices {
public:
    SimpleRenderer ();
    ~SimpleRenderer () { }

    // required methods:
    virtual bool get_matrix (Matrix44 &result, TransformationPtr xform,
                             float time);
    virtual bool get_matrix (Matrix44 &result, ustring from, float time);
    virtual bool get_matrix (Matrix44 &result, TransformationPtr xform);
    virtual bool get_matrix (Matrix44 &result, ustring from);
    virtual bool get_inverse_matrix (Matrix44 &result, ustring to, float time);

    virtual bool get_array_attribute (void *renderstate, bool derivatives, 
                                      ustring object, TypeDesc type, ustring name,
                                      int index, void *val );
    virtual bool get_attribute (void *renderstate, bool derivatives, ustring object,
                                TypeDesc type, ustring name, void *val);
    virtual bool get_userdata (bool derivatives, ustring name, TypeDesc type, 
                               void *renderstate, void *val);
    virtual bool has_userdata (ustring name, TypeDesc type, void *renderstate);

    // implementation specific method to setup internals:
    void name_transform (const char *name, const Matrix44& xform);
    void camera_params (const Matrix44& world_to_camera, float hfov, int xres, int yres);
                        
private:
    typedef std::map <ustring, Matrix44> TransformMap;
    TransformMap m_named_xforms;
    Matrix44 m_world_to_camera;
    float m_fov;
    int m_xres, m_yres;
};

OSL_NAMESPACE_EXIT
