# Macro to install targets to the appropriate locations.  Use this instead of
# the install(TARGETS ...) signature.
#
# Note: the original is in OIIO's src/cmake/oiio_macros.cmake
#
# Usage:
#
#    oiio_install_targets (target1 [target2 ...])
#
macro (oiio_install_targets)
    install (TARGETS ${ARGN}
             RUNTIME DESTINATION "${BIN_INSTALL_DIR}" COMPONENT user
             LIBRARY DESTINATION "${LIB_INSTALL_DIR}" COMPONENT user
             ARCHIVE DESTINATION "${LIB_INSTALL_DIR}" COMPONENT developer)
endmacro ()


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
    parse_arguments (_plugin "LINK_LIBRARIES" "" ${ARGN})
    set (_target_name ${CMAKE_CURRENT_SOURCE_DIR})
    # Get the name of the current directory and use it as the target name.
    get_filename_component (_target_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    add_library (${_target_name} SHARED ${_plugin_DEFAULT_ARGS})
    target_link_libraries (${_target_name} ${OPENIMAGEIO_LIBRARIES} ${_plugin_LINK_LIBRARIES})
    set_target_properties (${_target_name} PROPERTIES PREFIX "" FOLDER "Plugins")
    oiio_install_targets (${_target_name})
endmacro ()



add_oiio_plugin (oslinput.cpp
        LINK_LIBRARIES oslexec)
