# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required (VERSION 3.15)

set (OSL_VERSION "1.13.10.0")
set (OSL_VERSION_OVERRIDE "" CACHE STRING
     "Version override (use with caution)!")
mark_as_advanced (OSL_VERSION_OVERRIDE)
if (OSL_VERSION_OVERRIDE)
    set (OSL_VERSION ${OSL_VERSION_OVERRIDE})
endif ()

project (OSL VERSION ${OSL_VERSION}
         LANGUAGES CXX C
         HOMEPAGE_URL "https://github.com/AcademySoftwareFoundation/OpenShadingLanguage")
set (PROJ_NAME ${PROJECT_NAME})    # short name
string (TOLOWER ${PROJ_NAME} PROJ_NAME_LOWER)  # short name lower case
string (TOUPPER ${PROJ_NAME} PROJ_NAME_UPPER)  # short name upper case
set (PROJECT_VERSION_RELEASE_TYPE "" CACHE STRING
    "Build type, for example: dev, beta2, RC1 (empty string for normal release)")
set (${PROJECT_NAME}_VERSION_RELEASE_TYPE ${PROJECT_VERSION_RELEASE_TYPE})
set (PROJECT_AUTHORS "Contributors to the Open Shading Language project")
option (${PROJECT_NAME}_SUPPORTED_RELEASE
       "Set ON for supported release branch, OFF for master" ON)
if (${PROJECT_NAME}_SUPPORTED_RELEASE)
    set (${PROJECT_NAME}_DEV_RELEASE OFF)
else ()
    set (${PROJECT_NAME}_DEV_RELEASE ON)
endif ()

# Set PROJECT_IS_TOP_LEVEL to ON if if this is the top level project (not
# if this is included as a subproject of something else). Note that this is
# handled automatically for CMake >= 3.21.
if (CMAKE_VERSION VERSION_LESS 3.21)
    if ("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
        set (PROJECT_IS_TOP_LEVEL ON)
    endif ()
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

# If the user hasn't configured cmake with an explicit
# -DCMAKE_INSTALL_PREFIX=..., then set it to safely install into ./dist, to
# help prevent the user from accidentally writing over /usr/local or whatever.
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND PROJECT_IS_TOP_LEVEL)
    set (CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/dist" CACHE PATH
         "Installation location" FORCE)
endif()

message (STATUS "Configuring ${PROJECT_NAME} ${PROJECT_VERSION}")
message (STATUS "CMake ${CMAKE_VERSION}")
message (STATUS "CMake system           = ${CMAKE_SYSTEM}")
message (STATUS "CMake system name      = ${CMAKE_SYSTEM_NAME}")
message (STATUS "Project source dir     = ${PROJECT_SOURCE_DIR}")
message (STATUS "Project build dir      = ${CMAKE_BINARY_DIR}")
message (STATUS "Project install prefix = ${CMAKE_INSTALL_PREFIX}")
message (STATUS "Configuration types    = ${CMAKE_CONFIGURATION_TYPES}")
message (STATUS "Build type             = ${CMAKE_BUILD_TYPE}")

# Make the build area layout look a bit more like the final dist layout
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if ("${PROJECT_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message (FATAL_ERROR "Not allowed to run in-source build!")
endif ()


option (CMAKE_USE_FOLDERS "Use the FOLDER target property to organize targets into folders." ON)
mark_as_advanced (CMAKE_USE_FOLDERS)
if (CMAKE_USE_FOLDERS)
    set_property (GLOBAL PROPERTY USE_FOLDERS ON)
endif ()


# Version of the OSO file format and instruction set
set (OSO_FILE_VERSION_MAJOR 1)
set (OSO_FILE_VERSION_MINOR 0)


# This needs to be early, for CMAKE_INSTALL_FULL_DATADIR
include (GNUInstallDirs)


option (VERBOSE "Print lots of messages while compiling" OFF)
if (VERBOSE)
    set (CMAKE_MESSAGE_LOG_LEVEL "VERBOSE" CACHE STRING "CMake log level to display")
else ()
    set (CMAKE_MESSAGE_LOG_LEVEL "STATUS" CACHE STRING "CMake log level to display")
endif ()
set (OSL_LIBNAME_SUFFIX "" CACHE STRING
     "Optional name appended to ${PROJECT_NAME} libraries that are built")
option (OSL_BUILD_TESTS "Build the unit tests, testshade, testrender" ON)
if (WIN32)
    option (USE_LLVM_BITCODE "Generate embedded LLVM bitcode" OFF)
else ()
    option (USE_LLVM_BITCODE "Generate embedded LLVM bitcode" ON)
endif ()
option (OSL_BUILD_PLUGINS "Bool OSL plugins, for example OIIO plugin" ON)
option (OSL_BUILD_SHADERS "Build shaders" ON)
option (OSL_USE_OPTIX "Include OptiX support" OFF)
message(STATUS "OSL_USE_OPTIX: ${OSL_USE_OPTIX}")
set (OPTIX_EXTRA_LIBS CACHE STRING "Extra lib targets needed for OptiX")
set (CUDA_EXTRA_LIBS CACHE STRING "Extra lib targets needed for CUDA")
set (CUDA_TARGET_ARCH "sm_60" CACHE STRING "CUDA GPU architecture (e.g. sm_50)")
set (OSL_SHADER_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}/shaders"
     CACHE STRING "Directory where shaders will be installed")
set (OSL_PTX_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}/ptx"
     CACHE STRING "Directory where OptiX PTX files will be installed")
set (CMAKE_DEBUG_POSTFIX "" CACHE STRING "Library naming postfix for Debug builds (e.g., '_debug')")
option (OSL_USTRINGREP_IS_HASH "Always use ustringhash for strings" OFF)


set (OSL_NO_DEFAULT_TEXTURESYSTEM OFF CACHE BOOL "Do not use create a raw OIIO::TextureSystem")
if (OSL_NO_DEFAULT_TEXTURESYSTEM)
    add_definitions ("-DOSL_NO_DEFAULT_TEXTURESYSTEM=1")
endif ()

option (USE_FAST_MATH "Use fast math approximations (if no, then use system math library)" ON)
if (USE_FAST_MATH)
    add_definitions ("-DOSL_FAST_MATH=1")
else ()
    add_definitions ("-DOSL_FAST_MATH=0")
endif ()

option (OIIO_FMATH_SIMD_FRIENDLY "Make sure OIIO fmath functions are SIMD-friendly" OFF)
if (OIIO_FMATH_SIMD_FRIENDLY)
    add_definitions (-DOIIO_FMATH_SIMD_FRIENDLY=1)
endif ()

# Define OSL_INTERNAL symbol only when building OSL itself, will not be
# defined for downstream projects using OSL.
add_definitions (-DOSL_INTERNAL=1)

# To make sure we aren't relying on deprecated OIIO features, we define
# OIIO_DISABLE_DEPRECATED before including any OIIO headers.
add_definitions (-DOIIO_DISABLE_DEPRECATED=1)

# Set the default namespace. For symbol hiding reasons, it's important that
# the project name is a subset of the namespace name.
set (${PROJ_NAME}_NAMESPACE ${PROJECT_NAME} CACHE STRING
     "Customized outer namespace base name (version will be added)")
option (${PROJ_NAME}_NAMESPACE_INCLUDE_PATCH
        "Should the inner namespace include the patch number" ${${PROJECT_NAME}_DEV_RELEASE})
set (PROJ_NAMESPACE "${${PROJ_NAME}_NAMESPACE}")
string(REGEX MATCH ${PROJECT_NAME} NAMESPACE_HAS_PROJECT_NAME ${PROJ_NAMESPACE})
if (NOT NAMESPACE_HAS_PROJECT_NAME)
    set (PROJ_NAMESPACE ${PROJECT_NAME}_${PROJ_NAMESPACE})
endif ()
set (PROJ_NAMESPACE_V "${PROJ_NAMESPACE}_v${PROJECT_VERSION_MAJOR}_${PROJECT_VERSION_MINOR}")
if (${PROJ_NAME}_NAMESPACE_INCLUDE_PATCH)
    set (PROJ_NAMESPACE_V "${PROJ_NAMESPACE_V}_${PROJECT_VERSION_PATCH}")
endif ()
message(STATUS "Setting Namespace to: ${PROJ_NAMESPACE_V}")


list (APPEND CMAKE_MODULE_PATH
      "${PROJECT_SOURCE_DIR}/src/cmake/modules"
      "${PROJECT_SOURCE_DIR}/src/cmake")

# Helpful functions and macros for our project
include (colors)
include (check_is_enabled)
include (checked_find_package)

# All the C++ and compiler related options and adjustments
include (compiler)

# Utilities and options related to finding python and making python bindings
include (pythonutils)

include (externalpackages)
include (flexbison)
include (cuda_macros)
include (llvm_macros)

# Include all our testing apparatus and utils, but not if it's a subproject
if (${PROJECT_NAME}_BUILD_TESTS AND PROJECT_IS_TOP_LEVEL)
    include (testing)
endif ()



include_directories (
    BEFORE
    "${OSL_LOCAL_DEPS_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/include"
    "${CMAKE_BINARY_DIR}/src/include"
    "${CMAKE_BINARY_DIR}/include"
  )


# Make sure our math calculations are consistent,
# especially division.  Different compilers may have choose to use
# reciprocal division losing precision causing slightly different
# results which can lead to aliasing differences when running the testsuite
# Disable reciprocal division
if (CMAKE_COMPILER_IS_INTEL)
    if (MSVC)
        add_compile_options("/Qprec-div")
    else ()
        add_compile_options("-prec-div")
    endif ()
elseif (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_APPLECLANG OR CMAKE_COMPILER_IS_INTELCLANG)
    add_compile_options("-fno-reciprocal-math")
endif ()
       
# Tell CMake to process the sub-directories
add_subdirectory (src/include)
add_subdirectory (src/liboslcomp)
add_subdirectory (src/liboslquery)
add_subdirectory (src/liboslexec)
add_subdirectory (src/liboslnoise)
add_subdirectory (src/oslc)
add_subdirectory (src/oslinfo)

if (OSL_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory (src/testshade)
    add_subdirectory (src/testrender)
endif ()

if (OSL_BUILD_PLUGINS)
    add_subdirectory (src/osl.imageio)
endif ()

if (USE_QT AND (Qt5_FOUND OR Qt6_FOUND) AND NOT DISABLE_OSLTOY)
    add_subdirectory (src/osltoy)
endif ()

if (OSL_BUILD_SHADERS)
    add_subdirectory (src/shaders)
endif ()

option (INSTALL_DOCS "Install documentation" ON)
if (INSTALL_DOCS)
    add_subdirectory (src/doc)
endif ()

# Last minute site-specific instructions, if they exist
if (OSL_SITE AND EXISTS "${PROJECT_SOURCE_DIR}/site/${OSL_SITE}/cmake/sitecustom.cmake")
    include ("${PROJECT_SOURCE_DIR}/site/${OSL_SITE}/cmake/sitecustom.cmake")
endif ()

# install pkgconfig files
if ( NOT MSVC )
   configure_file(src/build-scripts/oslexec.pc.in "${CMAKE_BINARY_DIR}/oslexec.pc" @ONLY)
   configure_file(src/build-scripts/oslcomp.pc.in "${CMAKE_BINARY_DIR}/oslcomp.pc" @ONLY)
   configure_file(src/build-scripts/oslquery.pc.in "${CMAKE_BINARY_DIR}/oslquery.pc" @ONLY)
   install (FILES "${CMAKE_BINARY_DIR}/oslexec.pc"
                  "${CMAKE_BINARY_DIR}/oslcomp.pc"
                  "${CMAKE_BINARY_DIR}/oslquery.pc"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
            COMPONENT developer)
endif()



#########################################################################
# Export the configuration files. There are also library-specific config
# exports in the CMakeLists.txt of libOpenImageIO.
include (CMakePackageConfigHelpers)

# the file containing the exported targets
set (OSL_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets.cmake")
# the version file
set (OSL_VERSION_CONFIG "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
# the config file that is actually looked for by find_package
set (OSL_PROJECT_CONFIG "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
# where all these files will be installed
set (OSL_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# first generate the version file in the binary dir
write_basic_package_version_file (
        ${OSL_VERSION_CONFIG}
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

# generate the Targets file in the binary dir using the targets collected in
# OSL_EXPORTED_TARGETS each target is added to OSL_EXPORTED_TARGETS through
# the macro install_target().
export (EXPORT OSL_EXPORTED_TARGETS FILE "${CMAKE_BINARY_DIR}/${OSL_TARGETS_EXPORT_NAME}")

# generate the config file from the template in the binary dir
configure_package_config_file ("${PROJECT_SOURCE_DIR}/src/cmake/Config.cmake.in"
        "${OSL_PROJECT_CONFIG}"
        INSTALL_DESTINATION "${OSL_CONFIG_INSTALL_DIR}")

# generate the config file from the template in the binary dir
install (FILES "${OSL_PROJECT_CONFIG}" "${OSL_VERSION_CONFIG}"
        DESTINATION "${OSL_CONFIG_INSTALL_DIR}")

install (FILES src/cmake/llvm_macros.cmake DESTINATION cmake)

set (PERMISSION_FLAGS OWNER_EXECUTE OWNER_READ OWNER_WRITE
                      GROUP_EXECUTE GROUP_READ
                      WORLD_EXECUTE WORLD_READ)
install (FILES src/build-scripts/serialize-bc.py
         DESTINATION build-scripts
         PERMISSIONS ${PERMISSION_FLAGS})

# install targets files
install (EXPORT OSL_EXPORTED_TARGETS
        DESTINATION ${OSL_CONFIG_INSTALL_DIR}
        FILE ${OSL_TARGETS_EXPORT_NAME}
        NAMESPACE ${PROJECT_NAME}::)

if (PROJECT_IS_TOP_LEVEL AND BUILD_TESTING AND ${PROJECT_NAME}_BUILD_TESTS)
	osl_add_all_tests()
endif ()

if (PROJECT_IS_TOP_LEVEL)
    include (packaging)
endif ()
