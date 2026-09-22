// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "py_osl.h"

namespace PyOSL {

using namespace OSL;



void
declare_oslqueryparam(py_module& m)
{
    using Parameter = OSLQuery::Parameter;

    py::class_<Parameter>(m, "Parameter")
        .def(py::init<>())
        .def(py::init<const Parameter&>())
        .OSL_PY_PROP_RO("name",
                        [](const Parameter& p) {
                            return osl_py::str(p.name.string());
                        })
        // NOTE: this exposes an OIIO TypeDesc, and OSL's module never
        // registers that type -- it relies on OpenImageIO's Python module
        // having registered it. So reading `type` only works if OIIO's module
        // has been imported AND was built with the same binding framework as
        // this one (each framework has its own type registry and they can't
        // see each other's); otherwise it raises TypeError. Both backends
        // behave identically here, and this predates the nanobind work.
        // `type_name` gives the same information as a plain string with no
        // such coupling, and is what callers should prefer.
        .OSL_PY_RW("type", &Parameter::type)
        .OSL_PY_PROP_RW(
            "type_name",
            [](const Parameter& p) { return osl_py::str(p.type_name()); },
            [](Parameter& p, const std::string& t) { p.type_name(t); })
        .OSL_PY_RW("isoutput", &Parameter::isoutput)
        .OSL_PY_RW("varlenarray", &Parameter::varlenarray)
        .OSL_PY_RW("isstruct", &Parameter::isstruct)
        .OSL_PY_RW("isclosure", &Parameter::isclosure)
        .OSL_PY_PROP_RO(
            "value",
            [](const Parameter& p) {
                py::object result;
                if (p.type.basetype == TypeDesc::INT)
                    result = C_to_val_or_tuple(cspan<int>(p.idefault), p.type);
                else if (p.type.basetype == TypeDesc::FLOAT)
                    result = C_to_val_or_tuple(cspan<float>(p.fdefault),
                                               p.type);
                else if (p.type.basetype == TypeDesc::STRING)
                    result = C_to_val_or_tuple(cspan<ustring>(p.sdefault),
                                               p.type);
                else
                    result = py::none();
                return result;
            })
        .OSL_PY_PROP_RO(
            "spacename",
            [](const Parameter& p) {
                py::object result;
                if (p.spacename.size() > 1) {
                    TypeDesc t(TypeDesc::STRING);
                    result = C_to_val_or_tuple(cspan<ustring>(p.spacename), t);
                } else if (p.spacename.size() == 1) {
                    TypeDesc t(TypeDesc::STRING, p.spacename.size());
                    result = C_to_val_or_tuple(cspan<ustring>(p.spacename), t);
                } else {
                    result = py::none();
                }
                return result;
            })
        .OSL_PY_PROP_RO("fields",
                        [](const Parameter& p) {
                            py::object result;
                            if (p.isstruct) {
                                TypeDesc t(TypeDesc::STRING, p.fields.size());
                                result = C_to_val_or_tuple(cspan<ustring>(
                                                               p.fields),
                                                           t);
                            } else {
                                result = py::none();
                            }
                            return result;
                        })
        .OSL_PY_PROP_RO("structname",
                        [](const Parameter& p) {
                            return osl_py::str(p.structname.string());
                        })
        .OSL_PY_PROP_RO("metadata",
                        [](const Parameter& p) { return p.metadata; });
}



void
declare_oslquery(py_module& m)
{
    py::class_<OSLQuery>(m, "OSLQuery")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::string&>(), "shadername"_a,
             "searchpath"_a = "")

        //    OSLQuery (const ShaderGroup *group, int layernum)

        .def(
            "open",
            [](OSLQuery& self, const std::string& shadername,
               const std::string& searchpath) {
                return self.open(shadername, searchpath);
            },
            "shadername"_a, "searchpath"_a = "")
        .def(
            "open_bytecode",
            [](OSLQuery& self, const std::string& buffer) {
                return self.open_bytecode(buffer);
            },
            "buffer"_a)

        //    bool init (const ShaderGroup *group, int layernum);

        .def("shadertype",
             [](const OSLQuery& self) { return self.shadertype().string(); })
        .def("shadername",
             [](const OSLQuery& self) { return self.shadername().string(); })

        .OSL_PY_PROP_RO("nparams",
                        [](const OSLQuery& p) { return p.nparams(); })
        .OSL_PY_PROP_RO("parameters",
                        [](const OSLQuery& self) { return self.parameters(); })

        .OSL_PY_PROP_RO("metadata",
                        [](const OSLQuery& self) { return self.metadata(); })

        .def("__len__", [](const OSLQuery& p) { return p.nparams(); })
        .def("__getitem__",
             [](const OSLQuery& self, size_t i) {
                 auto p = self.getparam(i);
                 if (!p)
                     throw py::index_error();
                 return *p;
             })
        .def("__getitem__",
             [](const OSLQuery& self, const std::string& name) {
                 auto p = self.getparam(name);
                 if (!p)
                     osl_py::throw_key_error("parameter '" + name
                                             + "' does not exist");
                 return *p;
             })
        .def(
            "__iter__",
            [](const OSLQuery& self) {
                return osl_py::make_iterator<OSLQuery>(self.parameters().begin(),
                                                       self.parameters().end());
            },
            py::keep_alive<0, 1>())

        .def(
            "geterror",
            [](OSLQuery& self, bool clear_error) {
                return self.geterror(clear_error);
            },
            "clear_error"_a = true);
}



// Global (OSL scope) symbols
void
declare_module_attributes(py_module& m)
{
    m.attr("osl_version")    = OSL_VERSION;
    m.attr("VERSION")        = OSL_VERSION;
    m.attr("VERSION_STRING") = osl_py::str(OSL_LIBRARY_VERSION_STRING);
    m.attr("VERSION_MAJOR")  = OSL_VERSION_MAJOR;
    m.attr("VERSION_MINOR")  = OSL_VERSION_MINOR;
    m.attr("VERSION_PATCH")  = OSL_VERSION_PATCH;
    m.attr("INTRO_STRING")   = osl_py::str(OSL_INTRO_STRING);
    m.attr("__version__")    = osl_py::str(OSL_LIBRARY_VERSION_STRING);
}



#if defined(OSL_PY_BACKEND_NANOBIND)

}  // namespace PyOSL
// NB_MODULE, unlike PYBIND11_MODULE, must appear at global scope, so the
// namespace has to close before it rather than after.

// When both backends are built, this module is the second one and lives
// inside a package whose __init__.py re-exports it, so it needs a distinct
// name. When nanobind is the only backend, it is a drop-in replacement for
// the pybind11 module and takes the plain name.
#    if defined(OSL_PY_NANOBIND_ISOLATED_PACKAGE)
NB_MODULE(_oslquery, m)
#    else
NB_MODULE(oslquery, m)
#    endif
{
#    if PY_VERSION_HEX < 0x030a0000
    // Python 3.9 tears the interpreter down in an order that makes nanobind
    // report leaks that aren't there. Not an issue on 3.10+.
    // https://github.com/wjakob/nanobind/discussions/1405
    py::set_leak_warnings(false);
#    endif

    PyOSL::declare_module_attributes(m);
    PyOSL::declare_oslqueryparam(m);
    PyOSL::declare_oslquery(m);
}

#else

PYBIND11_MODULE(oslquery, m)
{
    declare_module_attributes(m);
    declare_oslqueryparam(m);
    declare_oslquery(m);
}

}  // namespace PyOSL

#endif
