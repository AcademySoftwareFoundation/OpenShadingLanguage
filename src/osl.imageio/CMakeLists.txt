# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# Macro to add a build target for an IO plugin.
#
# Note: the original is in OIIO's src/cmake/oiio_macros.cmake
#
# Usage:
#
# add_oiio_plugin ( source1 [source2 ...]
#                   [LINK_LIBRARIES external_lib1 ...] )
#
# The plugin name is deduced from the name of the current directory and the
# source is automatically linked against OpenImageIO.  Additional libraries
# (for example, libpng) may be specified after the  LINK_LIBRARIES
# keyword.
#
macro (add_oiio_plugin)
    cmake_parse_arguments (_plugin "" "NAME" "SRC;INCLUDE_DIRS;LINK_LIBRARIES;DEFINITIONS" ${ARGN})
       # Arguments: <prefix> <options> <one_value_keywords> <multi_value_keywords> args...
    get_filename_component (_plugin_name ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)
    if (NOT _plugin_NAME)
        # If NAME is not supplied, infer target name (and therefore the
        # executable name) from the directory name.
        get_filename_component (_plugin_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    endif ()
    # if (NOT _plugin_SRC)
    #     # If SRC is not supplied, assume local cpp files are its source.
    #     file (GLOB _plugin_SRC *.cpp)
    # endif ()

    # Get the name of the current directory and use it as the target name.
    get_filename_component (_plugin_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    add_library (${_plugin_name} MODULE ${_plugin_UNPARSED_ARGUMENTS})

    target_compile_definitions (${_plugin_name} PRIVATE
                                ${_plugin_DEFINITIONS})
    target_include_directories (${_plugin_name} PRIVATE ${_plugin_INCLUDE_DIRS})
    target_link_libraries (${_plugin_name}
                           PUBLIC  OpenImageIO::OpenImageIO
                           PRIVATE ${_plugin_LINK_LIBRARIES})
    set_target_properties (${_plugin_name}
                           PROPERTIES
                               PREFIX ""
                               FOLDER "Plugins")
    if (VISIBILITY_COMMAND)
        set_property (TARGET ${_plugin_name}
                      APPEND PROPERTY LINK_FLAGS ${VISIBILITY_MAP_COMMAND})
    endif ()
    install_targets (${_plugin_name})
endmacro ()


add_oiio_plugin (oslinput.cpp
        LINK_LIBRARIES oslexec)
