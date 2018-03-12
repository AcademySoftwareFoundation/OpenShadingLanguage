# Module to find LLVM
#
# This module defines the following variables:
#  LLVM_FOUND       - True if llvm found.
#  LLVM_VERSION     - Full LLVM version
#  LLVM_INCLUDES    - where to find llvm headers
#  LLVM_LIBRARIES   - List of LLVM libraries to link against
#  LLVM_SYSTEM_LIBRARIES - additional libraries needed by LLVM
#  LLVM_DIRECTORY   - If not already set, the root of the LLVM install
#  LLVM_LIB_DIR     - where to find llvm libs
#  LLVM_TARGETS     - List of available LLVM targets
#  CLANG_LIBRARIES  - list of libraries for clang components (optional,
#                        those may not be found)
#
# The following input symbols may be used to help guide the search:
#  LLVM_DIRECTORY   - the root of the LLVM installation (if custom)
#  LLVM_FIND_QUIETLY - if true, will suppress most console output
#  LLVM_STATIC      - if true, will prefer static LLVM libs to dynamic

# try to find llvm-config, with a specific version if specified
if (LLVM_DIRECTORY)
    set (LLVM_CONFIG_PATH_HINTS "${LLVM_DIRECTORY}/bin")
endif ()
list (APPEND LLVM_CONFIG_PATH_HINTS
        "/usr/local/opt/llvm/${LLVM_VERSION}/bin/"
        "/usr/local/opt/llvm/bin/")
find_program (LLVM_CONFIG
              NAMES llvm-config-${LLVM_VERSION} llvm-config
              HINTS ${LLVM_CONFIG_PATH_HINTS} NO_DEFAULT_PATH)
find_program (LLVM_CONFIG
              NAMES llvm-config-${LLVM_VERSION} llvm-config
              HINTS ${LLVM_CONFIG_PATH_HINTS})
if (NOT LLVM_FIND_QUIETLY)
    message (STATUS "Found llvm-config '${LLVM_CONFIG}'")
endif ()

if (NOT LLVM_DIRECTORY)
    execute_process (COMMAND ${LLVM_CONFIG} --prefix
           OUTPUT_VARIABLE LLVM_DIRECTORY
           OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

execute_process (COMMAND ${LLVM_CONFIG} --version
       OUTPUT_VARIABLE LLVM_VERSION
       OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND ${LLVM_CONFIG} --libdir
       OUTPUT_VARIABLE LLVM_LIB_DIR
       OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND ${LLVM_CONFIG} --includedir
       OUTPUT_VARIABLE LLVM_INCLUDES
       OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND ${LLVM_CONFIG} --targets-built
       OUTPUT_VARIABLE LLVM_TARGETS
       OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT ${LLVM_VERSION} VERSION_LESS 3.8)
    execute_process (COMMAND ${LLVM_CONFIG} --system-libs
                     OUTPUT_VARIABLE LLVM_SYSTEM_LIBRARIES
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    string (REPLACE " " ";" LLVM_SYSTEM_LIBRARIES "${LLVM_SYSTEM_LIBRARIES}")
else ()
    # Older LLVM did not have llvm-config --system-libs, but we know that
    # on Linux, we'll need curses.
    find_package (Curses)
    if (CURSES_FOUND)
        list (APPEND LLVM_SYSTEM_LIBRARIES ${CURSES_LIBRARIES})
    endif ()
endif ()

find_library ( LLVM_LIBRARY
               NAMES LLVM-${LLVM_VERSION} LLVM
               PATHS ${LLVM_LIB_DIR})
find_library ( LLVM_MCJIT_LIBRARY
               NAMES LLVMMCJIT
               PATHS ${LLVM_LIB_DIR})


foreach (COMPONENT clangFrontend clangDriver clangSerialization
                   clangParse clangSema clangAnalysis clangAST clangBasic
                   clangEdit clangLex)
    find_library ( _CLANG_${COMPONENT}_LIBRARY
                  NAMES ${COMPONENT}
                  PATHS ${LLVM_LIB_DIR})
    if (_CLANG_${COMPONENT}_LIBRARY)
        list (APPEND CLANG_LIBRARIES ${_CLANG_${COMPONENT}_LIBRARY})
    endif ()
endforeach ()


# if (NOT LLVM_LIBRARY)
#     execute_process (COMMAND ${LLVM_CONFIG} --libfiles engine
#                      OUTPUT_VARIABLE LLVM_LIBRARIES
#                      OUTPUT_STRIP_TRAILING_WHITESPACE)
# endif ()

# shared llvm library may not be available, this is not an error if we use LLVM_STATIC.
if ((LLVM_LIBRARY OR LLVM_LIBRARIES OR LLVM_STATIC) AND LLVM_INCLUDES AND LLVM_DIRECTORY AND LLVM_LIB_DIR)
  if (LLVM_STATIC)
    # if static LLVM libraries were requested, use llvm-config to generate
    # the list of what libraries we need, and substitute that in the right
    # way for LLVM_LIBRARY.
    execute_process (COMMAND ${LLVM_CONFIG} --libfiles
                     OUTPUT_VARIABLE LLVM_LIBRARIES
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
    string (REPLACE " " ";" LLVM_LIBRARIES "${LLVM_LIBRARIES}")
    set (LLVM_LIBRARY "")
  else ()
    set (LLVM_LIBRARIES "${LLVM_LIBRARY}")
  endif ()
endif ()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args (LLVM
    REQUIRED_VARS
        LLVM_INCLUDES
        LLVM_LIBRARIES
    VERSION_VAR LLVM_VERSION
  )

if (LLVM_FOUND)
    message (STATUS "LLVM version  = ${LLVM_VERSION}")
    if (NOT LLVM_FIND_QUIETLY)
        message (STATUS "LLVM dir        = ${LLVM_DIRECTORY}")
        message (STATUS "LLVM includes   = ${LLVM_INCLUDES}")
        message (STATUS "LLVM lib dir    = ${LLVM_LIB_DIR}")
        message (STATUS "LLVM libraries  = ${LLVM_LIBRARIES}")
        message (STATUS "LLVM sys libs   = ${LLVM_SYSTEM_LIBRARIES}")
        message (STATUS "Clang libraries = ${CLANG_LIBRARIES}")
    endif ()
else()
    message (STATUS "LLVM not found. Specify LLVM_DIRECTORY to locate it.")
endif()
