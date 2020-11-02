# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# Python-related options.
option (USE_PYTHON "Build the Python bindings" ON)
set (PYTHON_VERSION "" CACHE STRING "Target version of python to try to find")
option (PYLIB_INCLUDE_SONAME "If ON, soname/soversion will be set for Python module library" OFF)
option (PYLIB_LIB_PREFIX "If ON, prefix the Python module with 'lib'" OFF)
set (PYMODULE_SUFFIX "" CACHE STRING "Suffix to add to Python module init namespace")


# Find Python. This macro should only be called if python is required. If
# Python cannot be found, it will be a fatal error.
macro (find_python)
    if (NOT VERBOSE)
        set (PythonInterp_FIND_QUIETLY true)
        set (PythonLibs_FIND_QUIETLY true)
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
    checked_find_package (Python ${PYTHON_VERSION}
                          ${_req}
                          COMPONENTS Interpreter Development
                          PRINT Python_Development_FOUND Python_VERSION
                                Python_EXECUTABLE Python_LIBRARIES
                                Python_Development_FOUND Python_Interpreter_FOUND)

    # The version that was found may not be the default or user
    # defined one.
    set (PYTHON_VERSION_FOUND ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})

    # Give hints to subsequent pybind11 searching to ensure that it finds
    # exactly the same version that we found.
    set (PythonInterp_FIND_VERSION PYTHON_VERSION_FOUND)
    set (PythonInterp_FIND_VERSION_MAJOR ${Python_VERSION_MAJOR})

    if (NOT DEFINED PYTHON_SITE_DIR)
        set (PYTHON_SITE_DIR "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_FOUND}/site-packages")
    endif ()
    if (VERBOSE)
        message (STATUS "    Python site packages dir ${PYTHON_SITE_DIR}")
        message (STATUS "    Python to include 'lib' prefix: ${PYLIB_LIB_PREFIX}")
        message (STATUS "    Python to include SO version: ${PYLIB_INCLUDE_SONAME}")
    endif ()
endmacro()



###########################################################################
# pybind11

macro (setup_python_module)
    cmake_parse_arguments (lib "" "TARGET;MODULE;LIBS" "SOURCES" ${ARGN})
    # Arguments: <prefix> <options> <one_value_keywords> <multi_value_keywords> args...

    set (target_name ${lib_TARGET})

    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        # Seems to be a problem on some systems, with pybind11 and python headers
        set_property (SOURCE ${lib_SOURCES} APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-macro-redefined ")
    endif ()

    # Add the library itself
    add_library (${target_name} MODULE ${lib_SOURCES})

    # Declare the libraries it should link against
    target_link_libraries (${target_name}
                           PRIVATE
                                pybind11::pybind11
                                Python::Python
                                ${lib_LIBS} ${SANITIZE_LIBRARIES})
    if (APPLE)
        set_target_properties (${target_name} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif ()

    # Exclude the 'lib' prefix from the name
    if (NOT PYLIB_LIB_PREFIX)
        target_compile_definitions(${target_name}
                                   PRIVATE "PYMODULE_NAME=${lib_MODULE}")
        set_target_properties (${target_name} PROPERTIES
                               OUTPUT_NAME ${lib_MODULE}
                               PREFIX "")
    else ()
        target_compile_definitions(${target_name}
                                   PRIVATE "PYMODULE_NAME=Py${lib_MODULE}")
        set_target_properties (${target_name} PROPERTIES
                               OUTPUT_NAME Py${lib_MODULE}
                               PREFIX lib)
    endif ()

    if (PYLIB_INCLUDE_SONAME)
        if (VERBOSE)
            message(STATUS "Setting Py${lib_MODULE} SOVERSION to: ${SOVERSION}")
        endif ()
        set_target_properties(${target_name} PROPERTIES
            VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
            SOVERSION ${SOVERSION} )
    endif()

    if (WIN32)
        set_target_properties (${target_name} PROPERTIES
                               DEBUG_POSTFIX "_d"
                               SUFFIX ".pyd")
    endif()

    # In the build area, put it in lib/python so it doesn't clash with the
    # non-python libraries of the same name (which aren't prefixed by "lib"
    # on Windows).
    set_target_properties (${target_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python/site-packages
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python/site-packages
            )

    install (TARGETS ${target_name}
             RUNTIME DESTINATION ${PYTHON_SITE_DIR} COMPONENT user
             LIBRARY DESTINATION ${PYTHON_SITE_DIR} COMPONENT user)

endmacro ()

