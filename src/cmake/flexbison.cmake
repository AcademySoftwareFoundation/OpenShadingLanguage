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


# On Mac, prefer the Homebrew version of Bison over the older version from
# MacOS/xcode in /usr/bin, which seems to be too old for the reentrant
# parser directives we use. Only do this if there is no BISON_ROOT
# specifying a particular Bison to use.
if (APPLE AND EXISTS /usr/local/opt
        AND NOT BISON_ROOT AND NOT DEFINED ENV{BISON_ROOT})
    find_program(BISON_EXECUTABLE NAMES /usr/local/opt/bison/bin/bison
                 DOC "path to the bison executable")
endif()

checked_find_package (BISON 2.7 REQUIRED
                      PRINT BISON_EXECUTABLE)
checked_find_package (FLEX 2.3.35 REQUIRED
                      PRINT FLEX_EXECUTABLE)

if ( FLEX_EXECUTABLE AND BISON_EXECUTABLE )
    macro ( FLEX_BISON flexsrc bisonsrc prefix srclist compiler_headers )
        # mangle osoparse & oslparse symbols to avoid multiple library conflicts
        # XXX: This may be excessive now that OSL::pvt::ExtraArg is mangled into the function signature
        add_definitions(-D${prefix}parse=${PROJ_NAMESPACE_V}_${prefix}parse)

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
          COMMAND ${FLEX_EXECUTABLE} ${FB_WINCOMPAT} --prefix=${prefix} -o ${flexoutputcxx} "${CMAKE_CURRENT_SOURCE_DIR}/${flexsrc}"
          MAIN_DEPENDENCY ${flexsrc}
          DEPENDS ${${compiler_headers}}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
    endmacro ()
endif ()
