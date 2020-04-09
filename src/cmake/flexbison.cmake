# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

# LG's macros for using flex and bison.
# Merely including this file will require flex and bison to be found.
#
# The main purpose is to define the macro FLEX_BISON:
#   FLEX_BISON ( flexsrc bisonsrc srclist compiler_headers )
# where
#   flexsrc = the name of the .l file
#   bisonsrc = the name of the .y file
#   prefix = the language prefix
#   srclist = the name of the list of source files to which the macro will
#             add the flex- and bison-generated .cpp files
#   compiler_headers = the name of the list of headers that are dependencies
#             for the .y and .l files.


checked_find_package (BISON REQUIRED)
checked_find_package (FLEX REQUIRED)

if ( FLEX_EXECUTABLE AND BISON_EXECUTABLE )
    macro ( FLEX_BISON flexsrc bisonsrc prefix srclist compiler_headers )
        if (VERBOSE)
            message (STATUS "FLEX_BISON flex=${flexsrc} bison=${bisonsrc} prefix=${prefix}")
        endif ()
        get_filename_component ( bisonsrc_we ${bisonsrc} NAME_WE )
        set ( bisonoutputcxx "${CMAKE_CURRENT_BINARY_DIR}/${bisonsrc_we}.cpp" )
        set ( bisonoutputh "${CMAKE_CURRENT_BINARY_DIR}/${bisonsrc_we}.h" )

        get_filename_component ( flexsrc_we ${flexsrc} NAME_WE )
        set ( flexoutputcxx "${CMAKE_CURRENT_BINARY_DIR}/${flexsrc_we}.cpp" )
        set ( ${srclist} ${${srclist}} ${bisonoutputcxx} ${flexoutputcxx} )

        # Be really sure that we prefer the FlexLexer.h that comes with
        # the flex binary we're using, not some other one in the system.
        get_filename_component ( FLEX_UP ${FLEX_EXECUTABLE} PATH )
        get_filename_component ( FLEX_UP_UP ${FLEX_UP} PATH )
        set ( FLEX_INCLUDE_DIR "${FLEX_UP_UP}/include" )
        if (VERBOSE)
            message (STATUS "Flex include dir = ${FLEX_INCLUDE_DIR}")
        endif ()
        include_directories ( ${FLEX_INCLUDE_DIR} )

        # include_directories ( ${CMAKE_CURRENT_BINARY_DIR} )
        include_directories ( ${CMAKE_CURRENT_SOURCE_DIR} )
        add_custom_command ( OUTPUT ${bisonoutputcxx}
          COMMAND ${BISON_EXECUTABLE} -dv -p ${prefix} -o ${bisonoutputcxx} "${CMAKE_CURRENT_SOURCE_DIR}/${bisonsrc}"
          MAIN_DEPENDENCY ${bisonsrc}
          DEPENDS ${${compiler_headers}}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
        if ( WINDOWS )
            set ( FB_WINCOMPAT --wincompat )
        else ()
            set ( FB_WINCOMPAT )
        endif ()
        add_custom_command ( OUTPUT ${flexoutputcxx}
          COMMAND ${FLEX_EXECUTABLE} ${FB_WINCOMPAT} -o ${flexoutputcxx} "${CMAKE_CURRENT_SOURCE_DIR}/${flexsrc}"
          MAIN_DEPENDENCY ${flexsrc}
          DEPENDS ${${compiler_headers}}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
    endmacro ()
endif ()
