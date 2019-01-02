file (GLOB shader_headers "*.h")
file (GLOB shader_source "*.osl")

# FIXME -- is this really necessary?
if (${CMAKE_GENERATOR} MATCHES "(Visual Studio.*)")
    # Work around visual studio outputting oslc.exe in a subfolder
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../oslc/oslc"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/../oslc/${CMAKE_BUILD_TYPE}/oslc.exe" "${CMAKE_CURRENT_BINARY_DIR}/../oslc/"
        DEPENDS oslc)
endif ()


# Macro to compile a shader with oslc. Syntax is:
#   oslc_compile (OSL osl_source_file
#                 [ DEPENDS list_of_dependencies ]
#                 [ INCLUDE_DIRS list_of_include_dirs_for_oslc ]
#                 [ DEFINES list_of_extra_definitions_for_oslc ]
#                 [ OSO_FILE optional_oso_filename_override ]
#                 [ OSO_LIST list_to_append_oso_filename ] )
macro (oslc_compile)
    cmake_parse_arguments (_shader ""
                           "OSL;OSO_FILE;OSO_LIST"
                           "DEPENDS;INCLUDE_DIRS;DEFINES" ${ARGN})
    # ^^ syntax: prefix options one-arg-keywords multi-arg-keywords args
    set (oslfile ${_shader_OSL})
    if (_shader_OSO_FILE)
        set (osofile ${_shader_OSO_FILE})
    else ()
        get_filename_component ( oslsrc_we ${_shader_OSL} NAME_WE )
        set (osofile "${CMAKE_CURRENT_BINARY_DIR}/${oslsrc_we}.oso")
    endif ()
    if (VERBOSE)
        message (STATUS "oslc will make '${oslfile}'  ->  '${osofile}'")
    endif ()
    set (stdosl_header "${CMAKE_SOURCE_DIR}/src/shaders/stdosl.h")
    set (oslc_args -q ${_shader_DEFINES} "-I${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_incdir ${_shader_INCLUDE_DIRS})
        list (APPEND oslc_args "-I${_incdir}")
    endforeach ()
    list (APPEND oslc_args "-I${CMAKE_SOURCE_DIR}/src/shaders")
    add_custom_command (OUTPUT ${osofile}
        COMMAND oslc ${oslc_args} "${oslfile}" -o "${osofile}"
        MAIN_DEPENDENCY ${oslsrc}
        DEPENDS ${_shader_DEPENDS} "${stdosl_header}" oslc
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "oslc ${oslsrc_we}")
    if (_shader_OSO_LIST)
        list (APPEND ${_shader_OSO_LIST} ${osofile})
    endif ()
endmacro ()


foreach (_shadername ${shader_source})
    oslc_compile (OSL ${_shadername}
                  DEPENDS ${shader_headers}
                  INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
                  OSO_LIST shader_objs)
endforeach ()
#message (STATUS "all shader objs = ${shader_objs}")

add_custom_target (shaders ALL
                   DEPENDS ${shader_objs}
                   SOURCES ${shader_source} ${shader_headers})

install (FILES ${shader_headers} ${shader_source} ${shader_objs}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/shaders)
