# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# Python-related options.
set_option (USE_PYTHON "Build the Python bindings" ON)
set (PYTHON_VERSION "" CACHE STRING "Target version of python to try to find")
set_cache (OSL_PYTHON_BINDINGS_BACKEND ""
     "Which Python binding backend(s) to build: pybind11, nanobind, both, or empty to auto-select (nanobind when OIIO >= 3.2 and Python >= 3.10, else pybind11)" VERBOSE)
set_property (CACHE OSL_PYTHON_BINDINGS_BACKEND PROPERTY STRINGS
              "" pybind11 nanobind both)

# The user-facing backend selector is normalized, auto-resolved, and turned
# into booleans by osl_resolve_python_bindings_backend(), below. It must run
# *after* both OpenImageIO and Python have been found (find_python() calls it
# at that point): an empty selector picks nanobind when OpenImageIO is >= 3.2
# (older OIIO ships pybind11 python bindings, and mixing binding frameworks
# between the OIIO and OSL python modules does not work -- see INSTALL.md) and
# Python is >= 3.10 (nanobind's minimum), otherwise pybind11. nanobind itself
# need not be installed: the checked_find_package below builds it locally when
# it is missing. An explicit pybind11 / nanobind / both is honored as-is.
macro (osl_resolve_python_bindings_backend)
    # set_cache means this can also be set by an environment variable of the
    # same name, which is how CI drives it.
    string (TOLOWER "${OSL_PYTHON_BINDINGS_BACKEND}" OSL_PYTHON_BINDINGS_BACKEND)
    if (OSL_PYTHON_BINDINGS_BACKEND STREQUAL "")
        if (OpenImageIO_VERSION AND OpenImageIO_VERSION VERSION_GREATER_EQUAL 3.2
                AND Python3_VERSION AND Python3_VERSION VERSION_GREATER_EQUAL 3.10)
            set (OSL_PYTHON_BINDINGS_BACKEND "nanobind")
            message (STATUS "OSL_PYTHON_BINDINGS_BACKEND not set; auto-selected "
                     "'nanobind' (OpenImageIO ${OpenImageIO_VERSION}, Python ${Python3_VERSION})")
        else ()
            set (OSL_PYTHON_BINDINGS_BACKEND "pybind11")
            message (STATUS "OSL_PYTHON_BINDINGS_BACKEND not set; auto-selected "
                     "'pybind11' (nanobind needs OpenImageIO >= 3.2 and Python >= 3.10; "
                     "have ${OpenImageIO_VERSION} and ${Python3_VERSION})")
        endif ()
    endif ()
    if (NOT OSL_PYTHON_BINDINGS_BACKEND MATCHES "^(pybind11|nanobind|both)$")
        message (FATAL_ERROR
                 "OSL_PYTHON_BINDINGS_BACKEND must be one of: pybind11, nanobind, both")
    endif ()

    # Derive internal switches used by externalpackages.cmake, testing.cmake,
    # and the Python helper macros below.
    set (OSL_BUILD_PYTHON_PYBIND11 OFF)
    set (OSL_BUILD_PYTHON_NANOBIND OFF)
    if (OSL_PYTHON_BINDINGS_BACKEND STREQUAL "pybind11"
            OR OSL_PYTHON_BINDINGS_BACKEND STREQUAL "both")
        set (OSL_BUILD_PYTHON_PYBIND11 ON)
    endif ()
    if (OSL_PYTHON_BINDINGS_BACKEND STREQUAL "nanobind"
            OR OSL_PYTHON_BINDINGS_BACKEND STREQUAL "both")
        set (OSL_BUILD_PYTHON_NANOBIND ON)
    endif ()
endmacro ()

if (WIN32)
    set (PYLIB_LIB_TYPE SHARED CACHE STRING "Type of library to build for python module (MODULE or SHARED)")
else ()
    set (PYLIB_LIB_TYPE MODULE CACHE STRING "Type of library to build for python module (MODULE or SHARED)")
endif ()


# Find Python. This macro should only be called if python is required. If
# Python cannot be found, it will be a fatal error.
macro (find_python)
    if (NOT VERBOSE)
        set (PythonInterp3_FIND_QUIETLY true)
        set (PythonLibs3_FIND_QUIETLY true)
    endif ()

    # Attempt to find the desired version, but fall back to other
    # additional versions.
    unset (_req)
    if (USE_PYTHON)
        set (_req REQUIRED)
        if (PYTHON_VERSION)
            list (APPEND _req EXACT)
        endif ()
    endif ()
    checked_find_package (Python3 ${PYTHON_VERSION}
                          ${_req}
                          VERSION_MIN 3.9
                          COMPONENTS Interpreter Development
                          PRINT Python3_VERSION Python3_EXECUTABLE
                                Python3_LIBRARIES
                                Python3_Development_FOUND
                                Python3_Interpreter_FOUND )

    # OpenImageIO was found earlier in externalpackages.cmake and Python3 just
    # above -- both are needed to auto-select the Python binding backend from
    # an empty OSL_PYTHON_BINDINGS_BACKEND, so resolve it here, before the
    # nanobind-specific Python find and the pybind11/nanobind package finds
    # that follow (all of which key off OSL_BUILD_PYTHON_{PYBIND11,NANOBIND}).
    osl_resolve_python_bindings_backend ()

    if (OSL_BUILD_PYTHON_NANOBIND)
        # nanobind's CMake package expects the generic FindPython targets and
        # variables (Python::Module, Python_EXECUTABLE, etc.), not the
        # versioned Python3::* targets that the rest of OSL uses. Ask for the
        # exact version we just found so the two can't disagree.
        find_package (Python ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}
                      EXACT REQUIRED
                      COMPONENTS Interpreter Development)
    endif ()

    # The version that was found may not be the default or user
    # defined one.
    set (PYTHON_VERSION_FOUND ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

    # Give hints to subsequent pybind11 searching to ensure that it finds
    # exactly the same version that we found.
    set (PythonInterp3_FIND_VERSION PYTHON_VERSION_FOUND)
    set (PythonInterp3_FIND_VERSION_MAJOR ${Python3_VERSION_MAJOR})

    if (NOT DEFINED PYTHON_SITE_DIR)
        set (PYTHON_SITE_DIR "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_FOUND}/site-packages")
    endif ()
    message (VERBOSE "    Python version found ${PYTHON_VERSION_FOUND}")
    message (VERBOSE "    Python site packages dir ${PYTHON_SITE_DIR}")
endmacro()


# Help CMake locate nanobind when it was installed as a Python package (pip
# or Homebrew), which is the common case -- nanobind ships its CMake package
# config inside the Python package rather than in a place find_package()'s
# prefix search would look.
#
# This is a function (not a macro) deliberately: its early return must not
# escape into whatever file happens to include pythonutils.cmake and call
# this at file scope (a macro's return() would abort that entire caller file,
# silently skipping everything after it).
function (discover_nanobind_cmake_dir)
    # Cached from a previous configure. Trust it only if it still points to a
    # real nanobind install -- it may be stale if nanobind was uninstalled or
    # upgraded since the cache was written.
    if (nanobind_DIR AND EXISTS "${nanobind_DIR}/nanobind-config.cmake")
        return ()
    endif ()
    # Don't second-guess an explicit user hint.
    if (nanobind_ROOT OR "$ENV{nanobind_DIR}" OR "$ENV{nanobind_ROOT}")
        return ()
    endif ()
    if (NOT Python3_Interpreter_FOUND)
        return ()
    endif ()

    execute_process (
        COMMAND ${Python3_EXECUTABLE} -m nanobind --cmake_dir
        RESULT_VARIABLE _osl_nanobind_result
        OUTPUT_VARIABLE _osl_nanobind_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (_osl_nanobind_result EQUAL 0
            AND EXISTS "${_osl_nanobind_cmake_dir}/nanobind-config.cmake")
        message (VERBOSE "    Found nanobind CMake package via Python at ${_osl_nanobind_cmake_dir}")
        set (nanobind_DIR "${_osl_nanobind_cmake_dir}" CACHE PATH
             "Path to the nanobind CMake package" FORCE)
    endif ()
endfunction ()


###########################################################################
# pybind11

macro (setup_python_module)
    cmake_parse_arguments (lib "" "TARGET;MODULE" "SOURCES;LIBS" ${ARGN})
    # Arguments: <prefix> <options> <one_value_keywords> <multi_value_keywords> args...

    set (target_name ${lib_TARGET})

    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT ${CMAKE_COMPILER_ID} STREQUAL "Intel")
        # Seems to be a problem on some systems, with pybind11 and python headers
        set_property (SOURCE ${lib_SOURCES} APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-macro-redefined ")
    endif ()

    pybind11_add_module(${target_name} ${PYLIB_LIB_TYPE} ${lib_SOURCES})

#    # Add the library itself
#    add_library (${target_name} MODULE ${lib_SOURCES})
#
    # Declare the libraries it should link against
    target_link_libraries (${target_name}
                           PRIVATE ${lib_LIBS})

    set (_module_LINK_FLAGS "${VISIBILITY_MAP_COMMAND} ${EXTRA_DSO_LINK_ARGS}")
    if (UNIX AND NOT APPLE)
        # Hide symbols from any static dependent libraries embedded here.
        set (_module_LINK_FLAGS "${_module_LINK_FLAGS} -Wl,--exclude-libs,ALL")
    endif ()
    set_target_properties (${target_name} PROPERTIES LINK_FLAGS ${_module_LINK_FLAGS})

    # Exclude the 'lib' prefix from the name
    target_compile_definitions(${target_name}
                               PRIVATE "PYMODULE_NAME=${lib_MODULE}")
    set_target_properties (${target_name} PROPERTIES
                           OUTPUT_NAME ${lib_MODULE}
                           # PREFIX ""
                           )

#    if (WIN32)
#        set_target_properties (${target_name} PROPERTIES
#                               DEBUG_POSTFIX "_d"
#                               SUFFIX ".pyd")
#    endif()

    # In the build area, put it in lib/python so it doesn't clash with the
    # non-python libraries of the same name (which aren't prefixed by "lib"
    # on Windows).
    set_target_properties (${target_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python/site-packages
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python/site-packages
            )

    install (TARGETS ${target_name}
             RUNTIME DESTINATION ${PYTHON_SITE_DIR}/${lib_MODULE} COMPONENT user
             LIBRARY DESTINATION ${PYTHON_SITE_DIR}/${lib_MODULE} COMPONENT user)

    # COMPONENT user to match the TARGETS install above -- without it, a
    # component-filtered install produces a package directory holding the
    # extension module but no __init__.py, which can't be imported.
    install(FILES __init__.py DESTINATION ${PYTHON_SITE_DIR}/${lib_MODULE}
            COMPONENT user)

endmacro ()



###########################################################################
# nanobind
#
# Same job as setup_python_module() above, but building the module with
# nanobind instead of pybind11. Arguments are the same, with MODULE naming
# the *package* (as for pybind11); the extension module inside it is named
# by this macro, because that depends on whether we're the only backend:
#
#   backend=nanobind : <builddir>/lib/python/site-packages/oslquery.so
#                      installed to ${PYTHON_SITE_DIR}/oslquery/
#                      -- exactly where and what pybind11 would have put
#                      there, i.e. a drop-in replacement.
#
#   backend=both     : <builddir>/lib/python/nanobind/oslquery/_oslquery.so
#                      installed to ${PYTHON_SITE_DIR}/nanobind/oslquery/
#                      -- kept out of the way of the pybind11 module, which
#                      owns the ordinary location. `import oslquery` picks
#                      whichever one is on sys.path; a small __init__.py
#                      re-exports _oslquery so the import name is the same
#                      either way.
#
macro (setup_python_module_nanobind)
    cmake_parse_arguments (lib "" "TARGET;MODULE" "SOURCES;LIBS;PACKAGE_FILES" ${ARGN})

    set (target_name ${lib_TARGET})

    if (NOT COMMAND nanobind_add_module)
        discover_nanobind_cmake_dir ()
    endif ()

    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT ${CMAKE_COMPILER_ID} STREQUAL "Intel")
        # Seems to be a problem on some systems, with the python headers
        set_property (SOURCE ${lib_SOURCES} APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-macro-redefined ")
    endif ()

    # Note: unlike pybind11_add_module, this takes no MODULE/SHARED argument.
    nanobind_add_module (${target_name} ${lib_SOURCES})

    # nanobind's own sources trip -Wformat-nonliteral on clang.
    if (TARGET nanobind-static AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang"
                                    OR CMAKE_CXX_COMPILER_ID MATCHES "Apple"
                                    OR CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM"))
        target_compile_options (nanobind-static PUBLIC -Wno-format-nonliteral)
    endif ()

    target_link_libraries (${target_name} PRIVATE ${lib_LIBS})
    target_compile_definitions (${target_name} PRIVATE OSL_PY_BACKEND_NANOBIND)

    set (_module_LINK_FLAGS "${VISIBILITY_MAP_COMMAND} ${EXTRA_DSO_LINK_ARGS}")
    if (UNIX AND NOT APPLE)
        # Hide symbols from any static dependent libraries embedded here.
        set (_module_LINK_FLAGS "${_module_LINK_FLAGS} -Wl,--exclude-libs,ALL")
    endif ()
    set_target_properties (${target_name} PROPERTIES
                           LINK_FLAGS ${_module_LINK_FLAGS}
                           DEBUG_POSTFIX "")

    if (OSL_PYTHON_BINDINGS_BACKEND STREQUAL "both")
        set (_nanobind_build_dir ${CMAKE_BINARY_DIR}/lib/python/nanobind/${lib_MODULE})
        set (_nanobind_install_dir ${PYTHON_SITE_DIR}/nanobind/${lib_MODULE})
        target_compile_definitions (${target_name}
                                    PRIVATE OSL_PY_NANOBIND_ISOLATED_PACKAGE)
        set_target_properties (${target_name} PROPERTIES OUTPUT_NAME _${lib_MODULE})
    else ()
        set (_nanobind_build_dir ${CMAKE_BINARY_DIR}/lib/python/site-packages)
        set (_nanobind_install_dir ${PYTHON_SITE_DIR}/${lib_MODULE})
        set_target_properties (${target_name} PROPERTIES OUTPUT_NAME ${lib_MODULE})
    endif ()

    set_target_properties (${target_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${_nanobind_build_dir}
            ARCHIVE_OUTPUT_DIRECTORY ${_nanobind_build_dir}
            )

    install (TARGETS ${target_name}
             RUNTIME DESTINATION ${_nanobind_install_dir} COMPONENT user
             LIBRARY DESTINATION ${_nanobind_install_dir} COMPONENT user)

    if (OSL_PYTHON_BINDINGS_BACKEND STREQUAL "both")
        # The isolated package needs its own __init__.py (re-exporting
        # _${lib_MODULE}), and needs it in the build tree too, since that's
        # what the testsuite imports.
        configure_file (${lib_PACKAGE_FILES} ${_nanobind_build_dir}/__init__.py
                        COPYONLY)
        install (FILES ${lib_PACKAGE_FILES} DESTINATION ${_nanobind_install_dir}
                 COMPONENT user RENAME __init__.py)
    else ()
        # Drop-in replacement: same __init__.py the pybind11 module installs,
        # and like it, nothing extra in the build tree (the testsuite imports
        # the bare extension module from site-packages).
        install (FILES __init__.py DESTINATION ${_nanobind_install_dir}
                 COMPONENT user)
    endif ()

endmacro ()

