// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage



#include <cstdio>
#include <cstdlib>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/thread.h>
#include <OpenImageIO/typedesc.h>

#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>

using namespace OIIO;



OSL_NAMESPACE_BEGIN


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
/// Special options in the options list include:
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


class OSLInput final : public ImageInput {
public:
    OSLInput();
    virtual ~OSLInput() override;
    virtual const char* format_name(void) const override { return "osl"; }
    virtual int supports(string_view feature) const override
    {
        return (feature == "procedural");
    }
    virtual bool valid_file(const std::string& filename) const override;
    virtual bool open(const std::string& name, ImageSpec& newspec) override;
    virtual bool open(const std::string& name, ImageSpec& newspec,
                      const ImageSpec& config) override;
    virtual bool close() override;
    virtual int current_subimage(void) const override { return m_subimage; }
    virtual int current_miplevel(void) const override { return m_miplevel; }
    virtual bool seek_subimage(int subimage, int miplevel) override;
    virtual bool read_native_scanline(int subimage, int miplevel, int y, int z,
                                      void* data) override;
    virtual bool read_native_scanlines(int subimage, int miplevel, int ybegin,
                                       int yend, int z, void* data) override;
    virtual bool read_native_tile(int subimage, int miplevel, int x, int y,
                                  int z, void* data) override;
    virtual bool read_native_tiles(int subimage, int miplevel, int xbegin,
                                   int xend, int ybegin, int yend, int zbegin,
                                   int zend, void* data) override;

private:
    std::string m_filename;  ///< Stash the filename
    ShaderGroupRef m_group;
    std::vector<ustring> m_outputs;
    bool m_mip;
    int m_subimage, m_miplevel;
    ImageSpec m_topspec;  // spec of highest-res MIPmap

    // Reset everything to initial state
    void init()
    {
        m_group.reset();
        m_mip      = false;
        m_subimage = -1;
        m_miplevel = -1;
    }
};



// Obligatory material to make this a recognizable imageio plugin:
OIIO_PLUGIN_EXPORTS_BEGIN

OIIO_EXPORT ImageInput*
osl_input_imageio_create()
{
    return new OSLInput;
}

OIIO_EXPORT int osl_imageio_version = OIIO_PLUGIN_VERSION;

OIIO_EXPORT const char*
osl_imageio_library_version()
{
    return ustring(OSL_LIBRARY_VERSION_STRING).c_str();
}

OIIO_EXPORT const char* osl_input_extensions[] = { "osl", "oso", "oslgroup",
                                                   "oslbody", NULL };

OIIO_PLUGIN_EXPORTS_END



namespace pvt {


class OIIO_RendererServices final : public RendererServices {
public:
    OIIO_RendererServices(TextureSystem* texsys = NULL)
        : RendererServices(texsys)
    {
    }
    ~OIIO_RendererServices() {}

    int supports(string_view /*feature*/) const override { return false; }

    bool get_matrix(ShaderGlobals* /*sg*/, Matrix44& /*result*/,
                    TransformationPtr /*xform*/, float /*time*/) override
    {
        return false;  // FIXME?
    }
    bool get_matrix(ShaderGlobals* /*sg*/, Matrix44& /*result*/,
                    TransformationPtr /*xform*/) override
    {
        return false;  // FIXME?
    }
    bool get_matrix(ShaderGlobals* /*sg*/, Matrix44& /*result*/,
                    ustringhash /*from*/, float /*time*/) override
    {
        return false;  // FIXME?
    }
    bool get_matrix(ShaderGlobals* /*sg*/, Matrix44& /*result*/,
                    ustringhash /*from*/) override
    {
        return false;  // FIXME?
    }

    bool get_attribute(ShaderGlobals* /*sg*/, bool /*derivatives*/,
                       ustringhash /*object*/, TypeDesc /*type*/,
                       ustringhash /*name*/, void* /*val*/) override
    {
        return false;  // FIXME?
    }
    bool get_array_attribute(ShaderGlobals* /*sg*/, bool /*derivatives*/,
                             ustringhash /*object*/, TypeDesc /*type*/,
                             ustringhash /*name*/, int /*index*/,
                             void* /*val*/) override
    {
        return false;  // FIXME?
    }

    bool get_userdata(bool /*derivatives*/, ustringhash /*name*/,
                      TypeDesc /*type*/, ShaderGlobals* /*sg*/,
                      void* /*val*/) override
    {
        return false;  // FIXME?
    }
};



class ErrorRecorder final : public OIIO::ErrorHandler {
public:
    ErrorRecorder() : ErrorHandler() {}
    virtual void operator()(int errcode, const std::string& msg)
    {
        if (errcode >= EH_ERROR) {
            if (m_errormessage.size()
                && m_errormessage[m_errormessage.length() - 1] != '\n')
                m_errormessage += '\n';
            m_errormessage += msg;
        }
    }
    bool haserror() const { return m_errormessage.size(); }
    std::string geterror(bool erase = true)
    {
        std::string s;
        if (erase)
            std::swap(s, m_errormessage);
        else
            s = m_errormessage;
        return s;
    }

private:
    std::string m_errormessage;
};



static OIIO::mutex shading_mutex;
static ShadingSystem* shadingsys       = NULL;
static OIIO_RendererServices* renderer = NULL;
static ErrorRecorder errhandler;
static std::shared_ptr<TextureSystem> shared_texsys;


static void
setup_shadingsys()
{
    OIIO::lock_guard lock(shading_mutex);
    if (!shadingsys) {
#if OIIO_TEXTURESYSTEM_CREATE_SHARED
        if (!shared_texsys)
            shared_texsys = TextureSystem::create(true);
        auto ts = shared_texsys.get();
#else
        auto ts = TextureSystem::create(true);
#endif
        renderer   = new OIIO_RendererServices(ts);
        shadingsys = new ShadingSystem(renderer, NULL, &errhandler);
    }
}

}  // namespace pvt
using namespace pvt;



OSLInput::OSLInput()
{
    init();
}



OSLInput::~OSLInput()
{
    // Close, if not already done.
    close();
}



/// Deconstruct a "URI" string into the "fllename" part (returned) and turn
/// the "query" part into a series of pairs of id and value. For example,
///     deconstruct_uri("foo.tif?bar=1&blah=\"hello world\"", args)
/// would be expected to return "foo.tif" and *args would contain two
/// pairs: ("foo","1") and ("bar","\"hello world\"").
static string_view
deconstruct_uri(string_view uri,
                std::vector<std::pair<string_view, string_view>>* args = NULL)
{
    if (args)
        args->clear();
    size_t arg_start = uri.find('?');
    if (arg_start == string_view::npos)
        return uri;
    string_view argstring = uri.substr(arg_start + 1);
    string_view filename  = uri.substr(0, arg_start);
    if (!args)
        return filename;
    while (!argstring.empty()) {
        string_view id = Strutil::parse_until(argstring, "=&");
        string_view value;
        if (!id.size())
            break;
        if (!Strutil::parse_char(argstring, '=') || argstring.empty())
            break;
        if (argstring[0] == '\"')
            Strutil::parse_string(argstring, value, true, Strutil::KeepQuotes);
        else
            value = Strutil::parse_until(argstring, "&\t\r\n");
        args->push_back(std::make_pair(id, value));
        Strutil::parse_char(argstring, '&');
    }
    return filename;
}



bool
OSLInput::valid_file(const std::string& filename) const
{
    string_view shadername = deconstruct_uri(filename);
    if (!Strutil::ends_with(shadername, ".osl")
        && !Strutil::ends_with(shadername, ".oso")
        && !Strutil::ends_with(shadername, ".oslgroup")
        && !Strutil::ends_with(shadername, ".oslbody"))
        return false;
    return true;
}



bool
OSLInput::open(const std::string& name, ImageSpec& newspec)
{
    ImageSpec config;
    return open(name, newspec, config);
}



static void
parse_res(string_view res, int& x, int& y, int& z)
{
    if (Strutil::parse_int(res, x)) {
        if (Strutil::parse_char(res, 'x') && Strutil::parse_int(res, y)) {
            if (!(Strutil::parse_char(res, 'x') && Strutil::parse_int(res, z)))
                z = 1;
        } else {
            y = x;
            z = 1;
        }
    }
}



static bool
compile_buffer(const std::string& sourcecode, const std::string& shadername,
               std::string& errormessage)
{
    // std::cout << "source was\n---\n" << sourcecode << "---\n\n";
    errormessage.clear();
    std::string osobuffer;
    OSLCompiler compiler(&errhandler);
    std::vector<std::string> options;
    if (!compiler.compile_buffer(sourcecode, osobuffer, options)) {
        if (errhandler.haserror())
            errormessage = errhandler.geterror();
        else
            errormessage = Strutil::fmt::format("OSL: Could not compile \"{}\"",
                                                shadername);
        return false;
    }
    // std::cout << "Compiled to oso:\n---\n" << osobuffer << "---\n\n";
    if (!shadingsys->LoadMemoryCompiledShader(shadername, osobuffer)) {
        if (errhandler.haserror())
            errormessage = errhandler.geterror();
        else
            errormessage = Strutil::fmt::format(
                "OSL: Could not load compiled buffer from \"{}\"", shadername);
        return false;
    }
    return true;
}



// Add the attribute -- figure out the type
void
parse_param(string_view paramname, string_view val, ImageSpec& spec)
{
    TypeDesc type;  // start out unknown

    // If the param string starts with a type name, that's what it is
    if (size_t typeportion = type.fromstring(paramname)) {
        paramname.remove_prefix(typeportion);
        Strutil::skip_whitespace(paramname);
    }
    // If the value string starts with a type name, that's what it is
    else if (size_t typeportion = type.fromstring(val)) {
        val.remove_prefix(typeportion);
        Strutil::skip_whitespace(val);
    }

    if (type.basetype == TypeDesc::UNKNOWN) {
        // If we didn't find a type name, try to guess
        if (val.size() >= 2 && val.front() == '\"' && val.back() == '\"') {
            // Surrounded by quotes? it's a string (strip off the quotes)
            val.remove_prefix(1);
            val.remove_suffix(1);
            type = TypeString;
        } else if (Strutil::string_is<int>(val)) {
            // Looks like an int, is an int
            type = TypeInt;
        } else if (Strutil::string_is<float>(val)) {
            // Looks like a float, is a float
            type = TypeFloat;
        } else {
            // Everything else is assumed a string
            type = TypeString;
        }
    }

    // Read the values and set the attribute
    int n = type.numelements() * type.aggregate;
    if (type.basetype == TypeDesc::INT) {
        std::vector<int> values(n);
        for (int i = 0; i < n; ++i) {
            Strutil::parse_int(val, values[i]);
            Strutil::parse_char(val, ',');  // optional
        }
        if (n > 0)
            spec.attribute(paramname, type, &values[0]);
    }
    if (type.basetype == TypeDesc::FLOAT) {
        std::vector<float> values(n);
        for (int i = 0; i < n; ++i) {
            Strutil::parse_float(val, values[i]);
            Strutil::parse_char(val, ',');  // optional
        }
        if (n > 0)
            spec.attribute(paramname, type, &values[0]);
    } else if (type.basetype == TypeDesc::STRING) {
        std::vector<ustring> values(n);
        for (int i = 0; i < n; ++i) {
            string_view v;
            Strutil::parse_string(val, v);
            Strutil::parse_char(val, ',');  // optional
            values[i] = v;
        }
        if (n > 0)
            spec.attribute(paramname, type, &values[0]);
    }
}



bool
OSLInput::open(const std::string& name, ImageSpec& newspec,
               const ImageSpec& /*config*/)
{
    // std::cout << "OSLInput::open \"" << name << "\"\n";
    setup_shadingsys();

    std::vector<std::pair<string_view, string_view>> args;
    string_view shadername = deconstruct_uri(name, &args);
    if (shadername.empty())
        return false;
    if (!Strutil::ends_with(shadername, ".osl")
        && !Strutil::ends_with(shadername, ".oso")
        && !Strutil::ends_with(shadername, ".oslgroup")
        && !Strutil::ends_with(shadername, ".oslbody"))
        return false;

    m_filename = name;
    m_topspec  = ImageSpec(1024, 1024, 4, TypeDesc::FLOAT);

    // std::cout << "  name = " << shadername << " args? " << args.size() << "\n";
    for (size_t i = 0; i < args.size(); ++i) {
        // std::cout << "    " << args[i].first << "  =  " << args[i].second << "\n";
        if (args[i].first == "RES") {
            parse_res(args[i].second, m_topspec.width, m_topspec.height,
                      m_topspec.depth);
        } else if (args[i].first == "TILE" || args[i].first == "TILES") {
            parse_res(args[i].second, m_topspec.tile_width,
                      m_topspec.tile_height, m_topspec.tile_depth);
        } else if (args[i].first == "OUTPUT") {
            m_outputs.emplace_back(args[i].second);
        } else if (args[i].first == "MIP") {
            m_mip = Strutil::from_string<int>(args[i].second);
        } else if (args[i].first.size() && args[i].second.size()) {
            parse_param(args[i].first, args[i].second, m_topspec);
        }
    }
    if (m_outputs.empty()) {
        m_outputs.emplace_back("result");
        m_outputs.emplace_back("alpha");
    }

    m_topspec.full_x      = m_topspec.x;
    m_topspec.full_y      = m_topspec.y;
    m_topspec.full_z      = m_topspec.z;
    m_topspec.full_width  = m_topspec.width;
    m_topspec.full_height = m_topspec.height;
    m_topspec.full_depth  = m_topspec.depth;

    bool ok = true;
    if (Strutil::ends_with(shadername, ".oslgroup")) {  // Serialized group
        // No further processing necessary
        std::string groupspec;
        if (!OIIO::Filesystem::read_text_file(shadername, groupspec)) {
            // If it didn't name a disk file, assume it's the "inline"
            // serialized group.
            groupspec = groupspec.substr(0, groupspec.size() - 9);
        }
        // std::cout << "Processing group specification:\n---\n"
        //           << groupspec << "\n---\n";
        OIIO::lock_guard lock(shading_mutex);
        m_group = shadingsys->ShaderGroupBegin("", "surface", groupspec);
        if (!m_group)
            return false;  // Failed
        shadingsys->ShaderGroupEnd();
    }
    if (Strutil::ends_with(shadername, ".oso")) {  // Compiled shader
        OIIO::lock_guard lock(shading_mutex);
        shadername.remove_suffix(4);
        m_group = shadingsys->ShaderGroupBegin();
        for (auto&& pv : m_topspec.extra_attribs) {
            shadingsys->Parameter(pv.name(), pv.type(), pv.data(),
                                  pv.interp() == ParamValue::INTERP_CONSTANT);
        }
        if (!shadingsys->Shader("surface", shadername, "" /*layername*/)) {
            errorfmt("{}", errhandler.haserror() ? errhandler.geterror()
                                                 : std::string("OSL error"));
            ok = false;
        }
        shadingsys->ShaderGroupEnd();
    }

    if (Strutil::ends_with(shadername, ".osl")) {  // shader source
    }
    if (Strutil::ends_with(shadername, ".oslbody")) {  // shader source
        OIIO::lock_guard lock(shading_mutex);
        shadername.remove_suffix(8);
        static int exprcount   = 0;
        std::string exprname   = OIIO::Strutil::fmt::format("expr_{}",
                                                            exprcount++);
        std::string sourcecode = OIIO::Strutil::fmt::format(
            "shader {} (\n"
            "    float s = u [[ int interpolated=1 ]],\n"
            "    float t = v [[ int interpolated=1 ]],\n"
            "    output color result = 0,\n"
            "    output float alpha = 1,\n"
            "  )\n"
            "{{\n"
            "    {}\n"
            "    ;\n"
            "}}\n",
            exprname, shadername);
        // print("Expression-based shader text is:\n---\n{}\n---\n", sourcecode);
        std::string err;
        if (!compile_buffer(sourcecode, exprname, err)) {
            errorfmt("{}", err);
            return false;
        }
        m_group = shadingsys->ShaderGroupBegin();
        for (const auto& pv : m_topspec.extra_attribs) {
            shadingsys->Parameter(pv.name(), pv.type(), pv.data(),
                                  pv.interp() == ParamValue::INTERP_CONSTANT);
        }
        shadingsys->Shader("surface", exprname, "" /*layername*/);
        shadingsys->ShaderGroupEnd();
    }

    if (!ok || m_group.get() == NULL)
        return false;

    shadingsys->attribute(m_group.get(), "renderer_outputs",
                          TypeDesc(TypeDesc::STRING, m_outputs.size()),
                          &m_outputs[0]);

    ok &= seek_subimage(0, 0);
    if (ok)
        newspec = spec();
    else
        close();
    return ok;
}



bool
OSLInput::close()
{
    init();  // Reset to initial state
    return true;
}



bool
OSLInput::seek_subimage(int subimage, int miplevel)
{
    if (subimage == current_subimage() && miplevel == current_miplevel()) {
        return true;
    }

    if (subimage != 0)
        return false;  // We only make one subimage

    if (miplevel > 0 && !m_mip)
        return false;  // Asked for MIP levels but we aren't making them

    m_spec = m_topspec;
    for (m_miplevel = 0; m_miplevel < miplevel; ++m_miplevel) {
        if (m_spec.width == 1 && m_spec.height == 1 && m_spec.depth == 1)
            return false;  // Asked for more MIP levels than were available
        m_spec.width       = std::max(1, m_spec.width / 2);
        m_spec.height      = std::max(1, m_spec.height / 2);
        m_spec.depth       = std::max(1, m_spec.depth / 2);
        m_spec.full_width  = m_spec.width;
        m_spec.full_height = m_spec.height;
        m_spec.full_depth  = m_spec.depth;
    }
    return true;
}



bool
OSLInput::read_native_scanlines(int subimage, int miplevel, int ybegin,
                                int yend, int z, void* data)
{
    lock_guard lock(*this);
    if (!seek_subimage(subimage, miplevel))
        return false;

    if (!m_group.get()) {
        errorfmt("read_native_scanlines called with missing shading group");
        return false;
    }

    // Create an ImageBuf wrapper of the user's data
    ImageSpec spec = m_spec;  // Make a spec that describes just this scanline
    spec.y         = ybegin;
    spec.z         = z;
    spec.height    = yend - ybegin;
    spec.depth     = 1;
    ImageBuf ibwrapper(spec, data);

    // Now run the shader on the ImageBuf pixels, which really point to
    // the caller's data buffer.
    ROI roi(spec.x, spec.x + spec.width, spec.y, spec.y + spec.height, spec.z,
            spec.z + spec.depth);
    return shade_image(*shadingsys, *m_group, NULL, ibwrapper, m_outputs,
                       ShadePixelCenters, roi, 1);
}



bool
OSLInput::read_native_scanline(int subimage, int miplevel, int y, int z,
                               void* data)
{
    return read_native_scanlines(subimage, miplevel, y, y + 1, z, data);
}



bool
OSLInput::read_native_tiles(int subimage, int miplevel, int xbegin, int xend,
                            int ybegin, int yend, int zbegin, int zend,
                            void* data)
{
    lock_guard lock(*this);
    if (!seek_subimage(subimage, miplevel))
        return false;
    if (!m_group.get()) {
        errorfmt("read_native_tiles called with missing shading group");
        return false;
    }

    // Create an ImageBuf wrapper of the user's data
    ImageSpec spec = m_spec;  // Make a spec that describes just these tiles
    spec.x         = xbegin;
    spec.y         = ybegin;
    spec.z         = zbegin;
    spec.width     = xend - xbegin;
    spec.height    = yend - ybegin;
    spec.depth     = zend - zbegin;
    ImageBuf ibwrapper(spec, data);

    // Now run the shader on the ImageBuf pixels, which really point to
    // the caller's data buffer.
    ROI roi(spec.x, spec.x + spec.width, spec.y, spec.y + spec.height, spec.z,
            spec.z + spec.depth);
    return shade_image(*shadingsys, *m_group, NULL, ibwrapper, m_outputs,
                       ShadePixelCenters, roi, 1);
}



bool
OSLInput::read_native_tile(int subimage, int miplevel, int x, int y, int z,
                           void* data)
{
#if OIIO_PLUGIN_VERSION >= 24
    lock_guard lock(*this);
#else
    lock_guard lock(m_mutex);
#endif
    if (!seek_subimage(subimage, miplevel))
        return false;

    return read_native_tiles(
        subimage, miplevel, x,
        std::min(x + m_spec.tile_width, m_spec.x + m_spec.width), y,
        std::min(y + m_spec.tile_height, m_spec.y + m_spec.height), z,
        std::min(z + m_spec.tile_depth, m_spec.z + m_spec.depth), data);
}



OSL_NAMESPACE_END
