// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "py_osl.h"

#include <pybind11/embed.h>

namespace PyOSL {

using namespace OSL;



void
declare_oslqueryparam(py::module& m)
{
    using namespace pybind11::literals;
    using Parameter = OSLQuery::Parameter;

    py::class_<Parameter>(m, "Parameter")
        .def(py::init<>())
        .def(py::init<const Parameter&>())
        .def_property_readonly("name",
                               [](const Parameter& p) {
                                   return PY_STR(p.name.string());
                               })
        .def_readwrite("type", &Parameter::type)
        .def_readwrite("isoutput", &Parameter::isoutput)
        .def_readwrite("varlenarray", &Parameter::varlenarray)
        .def_readwrite("isstruct", &Parameter::isstruct)
        .def_readwrite("isclosure", &Parameter::isclosure)
        .def_readwrite("type", &Parameter::type)
        .def_property_readonly(
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
        .def_property_readonly(
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
        .def_property_readonly(
            "fields",
            [](const Parameter& p) {
                py::object result;
                if (p.isstruct) {
                    TypeDesc t(TypeDesc::STRING, p.fields.size());
                    result = C_to_val_or_tuple(cspan<ustring>(p.fields), t);
                } else {
                    result = py::none();
                }
                return result;
            })
        .def_property_readonly("structname",
                               [](const Parameter& p) {
                                   return PY_STR(p.structname.string());
                               })
        .def_property_readonly(
            "metadata", [](const Parameter& p) { return p.metadata; },
            py::return_value_policy::reference_internal);
}



void
declare_oslquery(py::module& m)
{
    using namespace pybind11::literals;

    py::class_<OSLQuery>(m, "OSLQuery")
        .def(py::init<>())
        .def(py::init([](const std::string& shadername,
                         const std::string& searchpath) {
                 return OSLQuery(shadername, searchpath);
             }),
             "shadername"_a, "searchpath"_a = "")

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

        .def_property_readonly("nparams",
                               [](const OSLQuery& p) { return p.nparams(); })
        .def_property_readonly(
            "parameters",
            [](const OSLQuery& self) { return self.parameters(); },
            py::return_value_policy::reference_internal)

        .def_property_readonly(
            "metadata", [](const OSLQuery& self) { return self.metadata(); },
            py::return_value_policy::reference_internal)

        .def("__len__", [](const OSLQuery& p) { return p.nparams(); })
        .def(
            "__getitem__",
            [](const OSLQuery& self, size_t i) {
                auto p = self.getparam(i);
                if (!p)
                    throw py::index_error();
                return *p;
            },
            py::return_value_policy::reference_internal)
        .def(
            "__getitem__",
            [](const OSLQuery& self, const std::string& name) {
                auto p = self.getparam(name);
                if (!p)
                    throw py::key_error("parameter '" + name
                                        + "' does not exist");
                return *p;
            },
            py::return_value_policy::reference_internal)
        .def(
            "__iter__",
            [](const OSLQuery& self) {
                return py::make_iterator(self.parameters().begin(),
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



// This OSL_DECLARE_PYMODULE mojo is necessary if we want to pass in the
// MODULE name as a #define. Google for Argument-Prescan for additional
// info on why this is necessary

#define OSL_DECLARE_PYMODULE(x) PYBIND11_MODULE(x, m)

OSL_DECLARE_PYMODULE(PYMODULE_NAME)
{
    // Force an OIIO module load so we have TypeDesc, among other things.
    py::module oiio = py::module::import("OpenImageIO");

    // Global (OSL scope) functions and symbols
    m.attr("osl_version")    = OSL_VERSION;
    m.attr("VERSION")        = OSL_VERSION;
    m.attr("VERSION_STRING") = PY_STR(OSL_LIBRARY_VERSION_STRING);
    m.attr("VERSION_MAJOR")  = OSL_VERSION_MAJOR;
    m.attr("VERSION_MINOR")  = OSL_VERSION_MINOR;
    m.attr("VERSION_PATCH")  = OSL_VERSION_PATCH;
    m.attr("INTRO_STRING")   = PY_STR(OSL_INTRO_STRING);
    m.attr("__version__")    = PY_STR(OSL_LIBRARY_VERSION_STRING);

    // Main OSL classes
    declare_oslqueryparam(m);
    declare_oslquery(m);
}

}  // namespace PyOSL
