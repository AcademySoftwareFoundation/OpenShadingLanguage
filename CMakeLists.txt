# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage

cmake_minimum_required (VERSION 3.12)
project (OSL VERSION 1.12.1.0
         LANGUAGES CXX C
         HOMEPAGE_URL "https://github.com/imageworks/OpenShadingLanguage")
set (PROJ_NAME ${PROJECT_NAME})    # short name
string (TOLOWER ${PROJ_NAME} PROJ_NAME_LOWER)  # short name lower case
string (TOUPPER ${PROJ_NAME} PROJ_NAME_UPPER)  # short name upper case
set (PROJECT_VERSION_RELEASE_TYPE "dev")   # "dev", "betaX", "RCY", ""
set (${PROJECT_NAME}_VERSION_RELEASE_TYPE ${PROJECT_VERSION_RELEASE_TYPE})
set (${PROJECT_NAME}_SUPPORTED_RELEASE 0)  # Change to 1 after release branch
set (PROJECT_AUTHORS "Contributors to the Open Shading Language project")

# Identify whether this is included as a subproject of something else
if (NOT "${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set (${PROJECT_NAME}_IS_SUBPROJECT ON)
    message (STATUS "${PROJECT_NAME} is configuring as a CMake subproject")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif ()

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
set (${PROJ_NAME}_NAMESPACE ${PROJECT_NAME} CACHE STRING
     "Customized outer namespace base name (version will be added)")
option (${PROJ_NAME}_NAMESPACE_INCLUDE_PATCH "Should the inner namespace include the patch number" OFF)
set (SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
     CACHE STRING "Set the SO version in the SO name of the output library")
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
option (OSL_BUILD_MATERIALX "Build MaterialX shaders" OFF)
option (USE_OPTIX "Include OptiX support" OFF)
set (OPTIX_EXTRA_LIBS CACHE STRING "Extra lib targets needed for OptiX")
set (CUDA_EXTRA_LIBS CACHE STRING "Extra lib targets needed for CUDA")
set (CUDA_TARGET_ARCH "sm_60" CACHE STRING "CUDA GPU architecture (e.g. sm_50)")
set (OSL_SHADER_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}/shaders"
     CACHE STRING "Directory where shaders will be installed")
set (OSL_PTX_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATADIR}/${PROJECT_NAME}/ptx"
     CACHE STRING "Directory where OptiX PTX files will be installed")
set (CMAKE_DEBUG_POSTFIX "" CACHE STRING "Library naming postfix for Debug builds (e.g., '_debug')")


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

if (USE_OPTIX)
    add_definitions ("-DOSL_USE_OPTIX=1")
endif ()


# Set the default namespace. For symbol hiding reasons, it's important that
# the project name is a subset of the namespace name.
set (PROJ_NAMESPACE "${${PROJ_NAME}_NAMESPACE}")
string(REGEX MATCH ${PROJECT_NAME} NAMESPACE_HAS_PROJECT_NAME ${PROJ_NAMESPACE})
if (NOT NAMESPACE_HAS_PROJECT_NAME)
    set (PROJ_NAMESPACE ${PROJECT_NAME}_${PROJ_NAMESPACE})
endif ()
set (PROJ_NAMESPACE_V "${PROJ_NAMESPACE}_v${PROJECT_VERSION_MAJOR}_${PROJECT_VERSION_MINOR}")
if (OIIO_NAMESPACE_INCLUDE_PATCH)
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

# Include all our testing apparatus and utils, but not if it's a subproject
if (${PROJECT_NAME}_BUILD_TESTS AND NOT ${PROJECT_NAME}_IS_SUBPROJECT)
    include (testing)
endif ()



include_directories (
    BEFORE
    "${OSL_LOCAL_DEPS_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/include"
    "${CMAKE_BINARY_DIR}/src/include"
    "${CMAKE_BINARY_DIR}/include"
  )


# Tell CMake to process the sub-directories
add_subdirectory (src/include)
add_subdirectory (src/liboslcomp)
add_subdirectory (src/liboslquery)
add_subdirectory (src/liboslexec)
add_subdirectory (src/liboslnoise)
add_subdirectory (src/oslc)
add_subdirectory (src/oslinfo)

if (OSL_BUILD_TESTS)
    add_subdirectory (src/testshade)
    add_subdirectory (src/testrender)
endif ()

if (OSL_BUILD_PLUGINS)
    add_subdirectory (src/osl.imageio)
endif ()

if (USE_QT AND Qt5_FOUND)
    add_subdirectory (src/osltoy)
endif ()

if (OSL_BUILD_SHADERS)
    add_subdirectory (src/shaders)
    if (OSL_BUILD_MATERIALX)
      add_subdirectory (src/shaders/MaterialX)
    endif ()
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

# install targets files
install (EXPORT OSL_EXPORTED_TARGETS
        DESTINATION ${OSL_CONFIG_INSTALL_DIR}
        FILE ${OSL_TARGETS_EXPORT_NAME}
        NAMESPACE ${PROJECT_NAME}::)




osl_add_all_tests()


if (NOT ${PROJECT_NAME}_IS_SUBPROJECT)
    include (packaging)
endif ()
