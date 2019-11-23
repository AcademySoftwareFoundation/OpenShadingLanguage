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
# (for example, libpng) may be specified after the optionl LINK_LIBRARIES
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
    if (OPENIMAGEIO_VERSION VERSION_LESS 2.1)
        # Prior to OIIO 2.1, OIIO built plugin modules as shared libs rather
        # than "modules". On OSX, that changed its extension from ".dylib"
        # to ".so". If building against OIIO < 2.1, do it the old way.
        # This clause can disappear when OIIO 2.1 is the minimum that OSL
        # supports.
        add_library (${_plugin_name} SHARED ${_plugin_UNPARSED_ARGUMENTS})
        set_target_properties (${_plugin_name}
                               PROPERTIES
                               VERSION ${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}
                               SOVERSION ${SOVERSION} )
    else ()
        add_library (${_plugin_name} MODULE ${_plugin_UNPARSED_ARGUMENTS})
    endif ()

    target_compile_definitions (${_plugin_name} PRIVATE
                                ${_plugin_DEFINITIONS})
    target_include_directories (${_plugin_name} PRIVATE ${_plugin_INCLUDE_DIRS}
                                                        ${OPENIMAGEIO_INCLUDES})
    target_link_libraries (${_plugin_name} PUBLIC ${OPENIMAGEIO_LIBRARIES}
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
