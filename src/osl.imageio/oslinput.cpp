/*
Copyright (c) 2009-2015 Sony Pictures Imageworks Inc., et al.
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





#include <cstdio>
#include <cstdlib>

#include <OpenImageIO/dassert.h>
#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/thread.h>
#include <OpenImageIO/imagebuf.h>

#include "OSL/oslexec.h"
#include "OSL/oslcomp.h"

using namespace OIIO;



OSL_NAMESPACE_ENTER


/// OSLInput is an ImageInput that behaves as if it's reading an image,
/// but actually it is executing OSL shaders to generate pixel values.
///
/// The filename is a "URI" form: shadername?opt1=val&opt2=val2...
///
/// The "shader name" may be any of:
///    name.oso         For a single osl shader, compiled to .oso
///    name.osl         For a single osl shader, still in source code
///                        (will be compiled in memory)
///    name.oslgroup    A file containing a serialized shader group
///    name.oslbody     Just the body of a simple OSL shader that writes
///                        to output params 'color result' and 'float alpha'
///                        which will be embedded in the surrounding
///                        boilerplate.
///
/// Speical options in the options list inclue:
///    RES=%dx%d        Set the resolution of the image (default: 1024x1024)
///    TILE=%dx%d       Set the tile size
///    MIP=%d           Should it generate all MIP levels (default: 0)
///    OUTPUT=%s        Name of output variable to use in the image
///                         (default: "result")
///
/// All other options are interpreted as setting shader parameters. The
/// format is "type name=value". If the type is omitted, it will be inferred
/// from the value (you get what you deserve if it's wrong). For aggregates
/// (arrays or triples), the value can be a comma-separated list. For
/// example:
///     "blah.oso?scale=2.0&octaves=3&point position=3.14,0,0"
///


class OSLInput : public ImageInput {
public:
    OSLInput ();
    virtual ~OSLInput ();
    virtual const char * format_name (void) const { return "osl"; }
#if OPENIMAGEIO_VERSION >= 10600
    virtual int supports (string_view feature) const {
        return (feature == "procedural");
    }
#else  /* Remove the following when OIIO <= 1.5 is no longer needed */
    virtual bool supports (const std::string& feature) const {
        return (feature == "procedural");
    }
#endif
    virtual bool valid_file (const std::string &filename) const;
    virtual bool open (const std::string &name, ImageSpec &newspec);
    virtual bool open (const std::string &name, ImageSpec &newspec,
                       const ImageSpec &config);
    virtual bool close ();
    virtual int current_subimage (void) const { return m_subimage; }
    virtual int current_miplevel (void) const { return m_miplevel; }
#if OIIO_PLUGIN_VERSION < 21   /* OIIO < 1.9 */
    virtual bool seek_subimage (int subimage, int miplevel, ImageSpec &newspec);
    virtual bool read_native_scanline (int y, int z, void *data);
    virtual bool read_native_scanlines (int ybegin, int yend, int z,
                                        void *data);
    virtual bool read_native_tile (int x, int y, int z, void *data);
    virtual bool read_native_tiles (int xbegin, int xend, int ybegin, int yend,
                                    int zbegin, int zend, void *data);
#else
    virtual bool seek_subimage (int subimage, int miplevel);
    virtual bool read_native_scanline (int subimage, int miplevel,
                                       int y, int z, void *data);
    virtual bool read_native_scanlines (int subimage, int miplevel,
                                        int ybegin, int yend, int z,
                                        void *data);
    virtual bool read_native_tile (int subimage, int miplevel,
                                   int x, int y, int z, void *data);
    virtual bool read_native_tiles (int subimage, int miplevel,
                                    int xbegin, int xend, int ybegin, int yend,
                                    int zbegin, int zend, void *data);
#endif
private:
    std::string m_filename;          ///< Stash the filename
    ShaderGroupRef m_group;
    std::vector<ustring> m_outputs;
    bool m_mip;
    int m_subimage, m_miplevel;
    ImageSpec m_topspec;   // spec of highest-res MIPmap

    // Reset everything to initial state
    void init () {
        m_group.reset ();
        m_mip = false;
        m_subimage = -1;
        m_miplevel = -1;
    }
};



// Obligatory material to make this a recognizeable imageio plugin:
OIIO_PLUGIN_EXPORTS_BEGIN

OIIO_EXPORT ImageInput *osl_input_imageio_create () { return new OSLInput; }

OIIO_EXPORT int osl_imageio_version = OIIO_PLUGIN_VERSION;

OIIO_EXPORT const char * osl_input_extensions[] = {
    "osl", "oso", "oslgroup", "oslbody", NULL
};

OIIO_PLUGIN_EXPORTS_END



namespace pvt {


class OIIO_RendererServices : public RendererServices {
public:
    OIIO_RendererServices (TextureSystem *texsys=NULL)
        : RendererServices (texsys) { }
    virtual ~OIIO_RendererServices () { }

    virtual int supports (string_view feature) const { return false; }

    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform, float time) {
        return false;   // FIXME?
    }
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform) {
        return false;   // FIXME?
    }
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from, float time) {
        return false;   // FIXME?
    }
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from) {
        return false;   // FIXME?
    }

    virtual bool get_attribute (ShaderGlobals *sg, bool derivatives,
                                ustring object, TypeDesc type, ustring name,
                                void *val) {
        return false;   // FIXME?
    }
    virtual bool get_array_attribute (ShaderGlobals *sg, bool derivatives,
                                      ustring object, TypeDesc type,
                                      ustring name, int index, void *val) {
        return false;   // FIXME?
    }

    virtual bool get_userdata (bool derivatives, ustring name, TypeDesc type,
                               ShaderGlobals *sg, void *val) {
        return false;   // FIXME?
    }
};



class ErrorRecorder : public OIIO::ErrorHandler {
public:
    ErrorRecorder () : ErrorHandler() { }
    virtual void operator () (int errcode, const std::string &msg) {
        if (errcode >= EH_ERROR) {
            if (m_errormessage.size() &&
                  m_errormessage[m_errormessage.length()-1] != '\n')
                m_errormessage += '\n';
            m_errormessage += msg;
        }
    }
    bool haserror () const { return m_errormessage.size(); }
    std::string geterror (bool erase=true) {
        std::string s;
        if (erase)
            std::swap (s, m_errormessage);
        else
            s = m_errormessage;
        return s;
    }
private:
    std::string m_errormessage;
};



static OIIO::mutex shading_mutex;
static ShadingSystem *shadingsys = NULL;
static OIIO_RendererServices *renderer = NULL;
static ErrorRecorder errhandler;



static void
setup_shadingsys ()
{
    OIIO::lock_guard lock (shading_mutex);
    if (! shadingsys) {
        renderer = new OIIO_RendererServices (TextureSystem::create(true));
        shadingsys = new ShadingSystem (renderer, NULL, &errhandler);
    }
}

}  // end pvt namespace
using namespace pvt;



OSLInput::OSLInput ()
{
    init ();
}



OSLInput::~OSLInput ()
{
    // Close, if not already done.
    close ();
}



/// Deconstruct a "URI" string into the "fllename" part (returned) and turn
/// the "query" part into a series of pairs of id and value. For example,
///     deconstruct_uri("foo.tif?bar=1&blah=\"hello world\"", args) 
/// would be expected to return "foo.tif" and *args would contain two
/// pairs: ("foo","1") and ("bar","\"hello world\"").
static string_view
deconstruct_uri (string_view uri,
                 std::vector<std::pair<string_view,string_view> > *args=NULL)
{
    if (args)
        args->clear ();
    size_t arg_start = uri.find ('?');
    if (arg_start == string_view::npos)
        return uri;
    string_view argstring = uri.substr (arg_start+1);
    string_view filename = uri.substr (0, arg_start);
    if (! args)
        return filename;
    while (! argstring.empty()) {
        string_view id = Strutil::parse_until (argstring, "=&");
        string_view value;
        if (! id.size())
            break;
        if (! Strutil::parse_char (argstring, '=') || argstring.empty())
            break;
        if (argstring[0] == '\"')
            Strutil::parse_string (argstring, value, true, Strutil::KeepQuotes);
        else
            value = Strutil::parse_until (argstring, "&\t\r\n");
        args->push_back (std::make_pair(id, value));
        Strutil::parse_char (argstring, '&');
    }
    return filename;
}



bool
OSLInput::valid_file (const std::string &filename) const
{
    string_view shadername = deconstruct_uri (filename);
    if (! Strutil::ends_with (shadername, ".osl") &&
        ! Strutil::ends_with (shadername, ".oso") &&
        ! Strutil::ends_with (shadername, ".oslgroup") &&
        ! Strutil::ends_with (shadername, ".oslbody"))
        return false;
    return true;
}



bool
OSLInput::open (const std::string &name, ImageSpec &newspec)
{
    ImageSpec config;
    return open (name, newspec, config);
}



static void
parse_res (string_view res, int &x, int &y, int &z)
{
    if (Strutil::parse_int (res, x)) {
        if (Strutil::parse_char (res, 'x') &&
            Strutil::parse_int (res, y)) {
            if (! (Strutil::parse_char(res, 'x') &&
                   Strutil::parse_int(res, z)))
                z = 1;
        } else {
            y = x;
            z = 1;
        }
    }
}



static bool
compile_buffer (const std::string &sourcecode,
                const std::string &shadername,
                std::string &errormessage)
{
    // std::cout << "source was\n---\n" << sourcecode << "---\n\n";
    errormessage.clear ();
    std::string osobuffer;
    OSLCompiler compiler (&errhandler);
    std::vector<std::string> options;
    if (! compiler.compile_buffer (sourcecode, osobuffer, options)) {
        if (errhandler.haserror())
            errormessage = errhandler.geterror();
        else
            errormessage = Strutil::format ("OSL: Could not compile \"%s\"", shadername);
        return false;
    }
    // std::cout << "Compiled to oso:\n---\n" << osobuffer << "---\n\n";
    if (! shadingsys->LoadMemoryCompiledShader (shadername, osobuffer)) {
        if (errhandler.haserror())
            errormessage = errhandler.geterror();
        else
            errormessage = Strutil::format ("OSL: Could not load compiled buffer from \"%s\"", shadername);
        return false;
    }
    return true;
}



// Add the attribute -- figure out the type
void
parse_param (string_view paramname, string_view val, ImageSpec &spec)
{
    TypeDesc type;   // start out unknown

    // If the param string starts with a type name, that's what it is
    if (size_t typeportion = type.fromstring (paramname)) {
        paramname.remove_prefix (typeportion);
        Strutil::skip_whitespace (paramname);
    }
    // If the value string starts with a type name, that's what it is
    else if (size_t typeportion = type.fromstring (val)) {
        val.remove_prefix (typeportion);
        Strutil::skip_whitespace (val);
    }

    if (type.basetype == TypeDesc::UNKNOWN) {
        // If we didn't find a type name, try to guess
        if (val.size() >= 2 && val.front() == '\"' && val.back() == '\"') {
            // Surrounded by quotes? it's a string (strip off the quotes)
            val.remove_prefix(1); val.remove_suffix(1);
            type = TypeDesc::TypeString;
        } else if (Strutil::string_is<int>(val)) {
            // Looks like an int, is an int
            type = TypeDesc::TypeInt;
        } else if (Strutil::string_is<float>(val)) {
            // Looks like a float, is a float
            type = TypeDesc::TypeFloat;
        } else {
            // Everything else is assumed a string
            type = TypeDesc::TypeString;
        }
    }

    // Read the values and set the attribute
    int n = type.numelements() * type.aggregate;
    if (type.basetype == TypeDesc::INT) {
        std::vector<int> values (n);
        for (int i = 0; i < n; ++i) {
            Strutil::parse_int (val, values[i]);
            Strutil::parse_char (val, ','); // optional
        }
        if (n > 0)
            spec.attribute (paramname, type, &values[0]);
    }
    if (type.basetype == TypeDesc::FLOAT) {
        std::vector<float> values (n);
        for (int i = 0; i < n; ++i) {
            Strutil::parse_float (val, values[i]);
            Strutil::parse_char (val, ','); // optional
        }
        if (n > 0)
            spec.attribute (paramname, type, &values[0]);
    } else if (type.basetype == TypeDesc::STRING) {
        std::vector<ustring> values (n);
        for (int i = 0; i < n; ++i) {
            string_view v;
            Strutil::parse_string (val, v);
            Strutil::parse_char (val, ','); // optional
            values[i] = v;
        }
        if (n > 0)
            spec.attribute (paramname, type, &values[0]);
    }
}



bool
OSLInput::open (const std::string &name, ImageSpec &newspec,
                const ImageSpec &config)
{
    // std::cout << "OSLInput::open \"" << name << "\"\n";
    setup_shadingsys ();

    std::vector<std::pair<string_view,string_view> > args;
    string_view shadername = deconstruct_uri (name, &args);
    if (shadername.empty())
        return false;
    if (! Strutil::ends_with (shadername, ".osl") &&
        ! Strutil::ends_with (shadername, ".oso") &&
        ! Strutil::ends_with (shadername, ".oslgroup") &&
        ! Strutil::ends_with (shadername, ".oslbody"))
        return false;

    m_filename = name;
    m_topspec = ImageSpec (1024, 1024, 4, TypeDesc::FLOAT);

    // std::cout << "  name = " << shadername << " args? " << args.size() << "\n";
    for (size_t i = 0; i < args.size(); ++i) {
        // std::cout << "    " << args[i].first << "  =  " << args[i].second << "\n";
        if (args[i].first == "RES") {
            parse_res (args[i].second, m_topspec.width, m_topspec.height, m_topspec.depth);
        } else if (args[i].first == "TILE" || args[i].first == "TILES") {
            parse_res (args[i].second, m_topspec.tile_width, m_topspec.tile_height,
                       m_topspec.tile_depth);
        } else if (args[i].first == "OUTPUT") {
            m_outputs.push_back (ustring(args[i].second));
        } else if (args[i].first == "MIP") {
            m_mip = Strutil::from_string<int>(args[i].second);
        } else if (args[i].first.size() && args[i].second.size()) {
            parse_param (args[i].first, args[i].second, m_topspec);
        }
    }
    if (m_outputs.empty()) {
        m_outputs.push_back (ustring("result"));
        m_outputs.push_back (ustring("alpha"));
    }

    m_topspec.full_x = m_topspec.x;
    m_topspec.full_y = m_topspec.y;
    m_topspec.full_z = m_topspec.z;
    m_topspec.full_width = m_topspec.width;
    m_topspec.full_height = m_topspec.height;
    m_topspec.full_depth = m_topspec.depth;

    bool ok = true;
    if (Strutil::ends_with (shadername, ".oslgroup")) { // Serialized group
        // No further processing necessary
        std::string groupspec;
        if (! OIIO::Filesystem::read_text_file (shadername, groupspec)) {
            // If it didn't name a disk file, assume it's the "inline"
            // serialized group.
            groupspec = groupspec.substr (0, groupspec.size()-9);
        }
        // std::cout << "Processing group specification:\n---\n"
        //           << groupspec << "\n---\n";
        OIIO::lock_guard lock (shading_mutex);
        m_group = shadingsys->ShaderGroupBegin ("", "surface", groupspec);
        if (! m_group)
            return false;   // Failed
        shadingsys->ShaderGroupEnd ();
    }
    if (Strutil::ends_with (shadername, ".oso")) { // Compiled shader
        OIIO::lock_guard lock (shading_mutex);
        shadername.remove_suffix (4);
        m_group = shadingsys->ShaderGroupBegin ();
        for (size_t p = 0, np = m_topspec.extra_attribs.size(); p < np; ++p) {
            const ParamValue &pv (m_topspec.extra_attribs[p]);
            shadingsys->Parameter (pv.name(), pv.type(), pv.data(),
                                   pv.interp() == ParamValue::INTERP_CONSTANT);
        }
        if (! shadingsys->Shader ("surface", shadername, "" /*layername*/ )) {
            error ("y %s", errhandler.haserror() ? errhandler.geterror() : std::string("OSL error"));
            ok = false;
        }
        shadingsys->ShaderGroupEnd ();
    }

    if (Strutil::ends_with (shadername, ".osl")) { // shader source
    }
    if (Strutil::ends_with (shadername, ".oslbody")) { // shader source
        OIIO::lock_guard lock (shading_mutex);
        shadername.remove_suffix (8);
        static int exprcount = 0;
        std::string exprname = OIIO::Strutil::format("expr_%d", exprcount++);
        std::string sourcecode =
            "shader " + exprname + " (\n"
            "    float s = u [[ int lockgeom=0 ]],\n"
            "    float t = v [[ int lockgeom=0 ]],\n"
            "    output color result = 0,\n"
            "    output float alpha = 1,\n"
            "  )\n"
            "{\n"
            "    " + std::string(shadername) + "\n"
            "    ;\n"
            "}\n";
        // std::cout << "Expression-based shader text is:\n---\n"
        //           << sourcecode << "---\n";
        std::string err;
        if (! compile_buffer (sourcecode, exprname, err)) {
            error ("%s", err);
            return false;
        }
        m_group = shadingsys->ShaderGroupBegin ();
        for (size_t p = 0, np = m_topspec.extra_attribs.size(); p < np; ++p) {
            const ParamValue &pv (m_topspec.extra_attribs[p]);
            shadingsys->Parameter (pv.name(), pv.type(), pv.data(),
                                   pv.interp() == ParamValue::INTERP_CONSTANT);
        }
        shadingsys->Shader ("surface", exprname, "" /*layername*/) ;
        shadingsys->ShaderGroupEnd ();
    }

    if (!ok || m_group.get() == NULL)
        return false;

    shadingsys->attribute (m_group.get(), "renderer_outputs",
                           TypeDesc(TypeDesc::STRING,m_outputs.size()),
                           &m_outputs[0]);

#if OIIO_PLUGIN_VERSION < 21
    return ok && seek_subimage (0, 0, newspec);
#else
    ok &= seek_subimage (0, 0);
    if (ok)
        newspec = spec();
    else
        close ();
    return ok;
#endif
}



bool
OSLInput::close ()
{
    init();  // Reset to initial state
    return true;
}



bool
OSLInput::seek_subimage (int subimage, int miplevel
#if OIIO_PLUGIN_VERSION < 21
                         , ImageSpec &newspec
#endif
                         )
{
    if (subimage == current_subimage() && miplevel == current_miplevel()) {
#if OIIO_PLUGIN_VERSION < 21
        newspec = spec();
#endif
        return true;
    }

    if (subimage != 0)
        return false;    // We only make one subimage

    if (miplevel > 0 && ! m_mip)
        return false;    // Asked for MIP levels but we aren't makign them

    m_spec = m_topspec;
    for (m_miplevel = 0; m_miplevel < miplevel; ++m_miplevel) {
        if (m_spec.width == 1 && m_spec.height == 1 && m_spec.depth == 1)
            return false;   // Asked for more MIP levels than were available
        m_spec.width = std::max (1, m_spec.width/2);
        m_spec.height = std::max (1, m_spec.height/2);
        m_spec.depth = std::max (1, m_spec.depth/2);
        m_spec.full_width = m_spec.width;
        m_spec.full_height = m_spec.height;
        m_spec.full_depth = m_spec.depth;
    }
#if OIIO_PLUGIN_VERSION < 21
    newspec = spec();
#endif
    return true;
}



bool
OSLInput::read_native_scanlines (
#if OIIO_PLUGIN_VERSION >= 21
                                 int subimage, int miplevel,
#endif
                                 int ybegin, int yend, int z, void *data)
{
#if OIIO_PLUGIN_VERSION >= 21
    lock_guard lock (m_mutex);
    if (! seek_subimage (subimage, miplevel))
        return false;
#endif

    // Create an ImageBuf wrapper of the user's data
    ImageSpec spec = m_spec; // Make a spec that describes just this scanline
    spec.y = ybegin;
    spec.z = z;
    spec.height = yend-ybegin;
    spec.depth = 1;
    ImageBuf ibwrapper (spec, data);

    // Now run the shader on the ImageBuf pixels, which really point to
    // the caller's data buffer.
    ASSERT (m_group.get());
    ROI roi (spec.x, spec.x+spec.width, spec.y, spec.y+spec.height,
             spec.z, spec.z+spec.depth);
    return shade_image (*shadingsys, *m_group, NULL, ibwrapper, m_outputs,
                        ShadePixelCenters, roi, 1);
}



bool
OSLInput::read_native_scanline (
#if OIIO_PLUGIN_VERSION >= 21
                                int subimage, int miplevel,
#endif
                                int y, int z, void *data)
{
#if OIIO_PLUGIN_VERSION >= 21
    return read_native_scanlines (subimage, miplevel, y, y+1, z, data);
#else
    return read_native_scanlines (y, y+1, z, data);
#endif
}



bool
OSLInput::read_native_tiles (
#if OIIO_PLUGIN_VERSION >= 21
                             int subimage, int miplevel,
#endif
                             int xbegin, int xend, int ybegin, int yend,
                             int zbegin, int zend, void *data)
{
#if OIIO_PLUGIN_VERSION >= 21
    lock_guard lock (m_mutex);
    if (! seek_subimage (subimage, miplevel))
        return false;
#endif

    // Create an ImageBuf wrapper of the user's data
    ImageSpec spec = m_spec; // Make a spec that describes just this scanline
    spec.x = xbegin;
    spec.y = ybegin;
    spec.z = zbegin;
    spec.width  = xend-xbegin;
    spec.height = yend-ybegin;
    spec.depth  = zend-zbegin;
    ImageBuf ibwrapper (spec, data);

    // Now run the shader on the ImageBuf pixels, which really point to
    // the caller's data buffer.
    ASSERT (m_group.get());
    ROI roi (spec.x, spec.x+spec.width, spec.y, spec.y+spec.height,
             spec.z, spec.z+spec.depth);
    return shade_image (*shadingsys, *m_group, NULL, ibwrapper, m_outputs,
                        ShadePixelCenters, roi, 1);
}



bool
OSLInput::read_native_tile (
#if OIIO_PLUGIN_VERSION >= 21
                            int subimage, int miplevel,
#endif
                            int x, int y, int z, void *data)
{
#if OIIO_PLUGIN_VERSION >= 21
    lock_guard lock (m_mutex);
    if (! seek_subimage (subimage, miplevel))
        return false;
#endif

    return
        read_native_tiles (
#if OIIO_PLUGIN_VERSION >= 21
                           subimage, miplevel,
#endif
                           x, std::min (x+m_spec.tile_width, m_spec.x+m_spec.width),
                           y, std::min (y+m_spec.tile_height, m_spec.y+m_spec.height),
                           z, std::min (z+m_spec.tile_depth, m_spec.z+m_spec.depth),
                           data);
}



OSL_NAMESPACE_EXIT
