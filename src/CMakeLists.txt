project (OSL)
set (OSL_VERSION_MAJOR 0)
set (OSL_VERSION_MINOR 1)
set (OSL_VERSION_PATCH 0)

CMAKE_MINIMUM_REQUIRED (VERSION 2.6)
SET (CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
message (STATUS "Project source dir = ${PROJECT_SOURCE_DIR}")
message (STATUS "Project build dir = ${CMAKE_BINARY_DIR}")

if (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message (FATAL_ERROR "Not allowed to run in-source build!")
endif ()


if (NOT CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE "Release")
endif ()

set(CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/cmake/modules"
    "${PROJECT_SOURCE_DIR}/cmake")
include (util_macros)
include (platform)
include (externalpackages)
include (flexbison)
include (oiio)
include_directories (${OPENIMAGEIO_INCLUDES})


include_directories ("include")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions ("-DDEBUG=1")
endif ()

# use, i.e. don't skip the full RPATH for the build tree
set (CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
set (CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
# the RPATH to be used when installing
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message (STATUS "CMAKE_INSTALL_RPATH = ${CMAKE_INSTALL_RPATH}")

# Tell CMake to process the sub-directories
add_subdirectory (liboslcomp)
add_subdirectory (liboslquery)
add_subdirectory (liboslexec)

add_subdirectory (oslc)
add_subdirectory (oslinfo)

add_subdirectory (include)
add_subdirectory (doc)



#########################################################################
# Testing
include (CTest)
add_subdirectory (testshade)
# Ugh, I can't seem to ADD_SUBDIRECTORY the testsuite dir if it's not a true
# subdirectory of source, so we fake it by creating a link.  FIXME!
exec_program ("cmake -E create_symlink ${PROJECT_SOURCE_DIR}/../testsuite ${PROJECT_SOURCE_DIR}/testsuite")
add_subdirectory (testsuite)


#########################################################################
# Packaging
set (CPACK_PACKAGE_VERSION_MAJOR ${OSL_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${OSL_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${OSL_VERSION_PATCH})
# "Vendor" is only used in copyright notices, so we use the same thing that
# the rest of the copyright notices say.
set (CPACK_PACKAGE_VENDOR "Sony Pictures Imageworks")
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenShadingLanguage is...")
set (CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/doc/Description.txt")
set (CPACK_PACKAGE_FILE_NAME OSL-${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}-${platform})
#set (CPACK_PACKAGE_INSTALL_DIRECTORY ${PROJECT_SOURCE_DIR}/..)
exec_program ("cmake -E copy ${PROJECT_SOURCE_DIR}/../LICENSE ${CMAKE_BINARY_DIR}/License.txt")
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/License.txt")
exec_program ("cmake -E copy ${PROJECT_SOURCE_DIR}/../README ${CMAKE_BINARY_DIR}/Readme.txt")
set (CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/Readme.txt")
set (CPACK_RESOURCE_FILE_WELCOME "${PROJECT_SOURCE_DIR}/doc/Welcome.txt")
#set (CPACK_PACKAGE_EXECUTABLES I'm not sure what this is for)
#set (CPACK_STRIP_FILES Do we need this?)
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set (CPACK_GENERATOR "TGZ;STGZ;RPM;DEB")
    set (CPACK_SOURCE_GENERATOR "TGZ")
endif ()
if (APPLE)
    set (CPACK_GENERATOR "TGZ;STGZ;PackageMaker")
    set (CPACK_SOURCE_GENERATOR "TGZ")
endif ()
set (CPACK_SOURCE_PACKAGE_FILE_NAME OSL-${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}-source)
#set (CPACK_SOURCE_STRIP_FILES ...FIXME...)
set (CPACK_SOURCE_IGNORE_FILES ".*~")
include (CPack)
