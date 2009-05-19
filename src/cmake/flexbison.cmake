# LG's macros for using flex and bison.
# Merely including this file will set FLEX_EXECUTABLE and BISON_EXECUTABLE
# to the locations of 'flex' and 'bison' on the system, and leave them
# undefined if those programs were not found.
# If both are found, there will also be a macro defined:
#   FLEX_BISON ( flexsrc bisonsrc srclist compiler_headers )
# where
#   flexsrc = the name of the .l file
#   bisonsrc = the name of the .y file
#   prefix = the language prefix
#   srclist = the name of the list of source files to which the macro will
#             add the flex- and bison-generated .cpp files
#   compiler_headers = the name of the list of headers that are dependencies
#             for the .y and .l files.


IF (NOT FLEX_EXECUTABLE)
  MESSAGE (STATUS "Looking for flex")
  FIND_PROGRAM (FLEX_EXECUTABLE flex)
  IF (FLEX_EXECUTABLE)
    MESSAGE (STATUS "Looking for flex -- ${FLEX_EXECUTABLE}")
  ENDIF (FLEX_EXECUTABLE)
ENDIF (NOT FLEX_EXECUTABLE) 

IF (NOT BISON_EXECUTABLE)
  MESSAGE (STATUS "Looking for bison")
  FIND_PROGRAM (BISON_EXECUTABLE bison)
  IF (BISON_EXECUTABLE)
    MESSAGE (STATUS "Looking for bison -- ${BISON_EXECUTABLE}")
  ENDIF (BISON_EXECUTABLE)
ENDIF (NOT BISON_EXECUTABLE)


IF ( FLEX_EXECUTABLE AND BISON_EXECUTABLE )
    MACRO ( FLEX_BISON flexsrc bisonsrc prefix srclist compiler_headers )
        MESSAGE (STATUS "FLEX_BISON flex=${flexsrc} bison=${bisonsrc} prefix=${prefix}")
        #MESSAGE (STATUS "  src ${CMAKE_CURRENT_SOURCE_DIR}")
        #MESSAGE (STATUS "  bin ${CMAKE_CURRENT_BINARY_DIR}")
        GET_FILENAME_COMPONENT ( bisonsrc_we ${bisonsrc} NAME_WE )
        SET ( bisonoutputcxx "${CMAKE_CURRENT_BINARY_DIR}/${bisonsrc_we}.cpp" )
        SET ( bisonoutputh "${CMAKE_CURRENT_BINARY_DIR}/${bisonsrc_we}.h" )
        # MESSAGE (STATUS "  bison output ${bisonoutputcxx} ${bisonoutputh}")

        GET_FILENAME_COMPONENT ( flexsrc_we ${flexsrc} NAME_WE )
        SET ( flexoutputcxx "${CMAKE_CURRENT_BINARY_DIR}/${flexsrc_we}.cpp" )
        # MESSAGE (STATUS "  flex output ${flexoutputcxx}")

        SET ( ${srclist} ${${srclist}} ${bisonoutputcxx} ${flexoutputcxx} )
        # MESSAGE (STATUS "  src list now ${${srclist}}")
        # MESSAGE (STATUS "  compiler headers = ${${compiler_headers}}")
        INCLUDE_DIRECTORIES ( ${CMAKE_CURRENT_BINARY_DIR} )
        INCLUDE_DIRECTORIES ( ${CMAKE_CURRENT_SOURCE_DIR} )
        ADD_CUSTOM_COMMAND ( OUTPUT ${bisonoutputcxx} 
          COMMAND ${BISON_EXECUTABLE} -dv -p ${prefix} -o ${bisonoutputcxx} ${CMAKE_CURRENT_SOURCE_DIR}/${bisonsrc}
          MAIN_DEPENDENCY ${bisonsrc}
          DEPENDS ${${compiler_headers}}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
        ADD_CUSTOM_COMMAND ( OUTPUT ${flexoutputcxx} 
          COMMAND ${FLEX_EXECUTABLE} -+ -o ${flexoutputcxx} ${CMAKE_CURRENT_SOURCE_DIR}/${flexsrc} 
          MAIN_DEPENDENCY ${flexsrc}
          DEPENDS ${${compiler_headers}}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
      ENDMACRO ()
ENDIF ()
