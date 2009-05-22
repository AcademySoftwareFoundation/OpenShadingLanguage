PROJECT (OSL)
SET ( OSL_VERSION_MAJOR 0 )
SET ( OSL_VERSION_MINOR 1 )
SET ( OSL_VERSION_PATCH 0 )

CMAKE_MINIMUM_REQUIRED (VERSION 2.6)
SET (CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
MESSAGE ( STATUS "Project source dir = ${PROJECT_SOURCE_DIR}" )
MESSAGE ( STATUS "Project build dir = ${CMAKE_BINARY_DIR}" )

IF ( ${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
    MESSAGE ( FATAL_ERROR "Not allowed to run in-source build!" )
ENDIF ()


IF ( NOT CMAKE_BUILD_TYPE ) 
    SET( CMAKE_BUILD_TYPE "Release" ) 
ENDIF ()

SET(CMAKE_MODULE_PATH
	"${PROJECT_SOURCE_DIR}/cmake/modules"
	"${PROJECT_SOURCE_DIR}/cmake")
INCLUDE ( util_macros )
INCLUDE ( platform )
INCLUDE ( externalpackages )
INCLUDE ( flexbison )
INCLUDE ( oiio )
INCLUDE_DIRECTORIES ( ${OPENIMAGEIO_INCLUDES} )


INCLUDE_DIRECTORIES ("include")

IF ( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    ADD_DEFINITIONS ( "-DDEBUG=1" )
ENDIF ()

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
# the RPATH to be used when installing
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

MESSAGE (STATUS "CMAKE_INSTALL_RPATH = ${CMAKE_INSTALL_RPATH}")

# Tell CMake to process the sub-directories
ADD_SUBDIRECTORY ( liboslcomp )
ADD_SUBDIRECTORY ( liboslquery )
ADD_SUBDIRECTORY ( liboslexec )

ADD_SUBDIRECTORY ( oslc )
ADD_SUBDIRECTORY ( oslinfo )

ADD_SUBDIRECTORY ( include )
ADD_SUBDIRECTORY ( doc )



#########################################################################
# Testing
INCLUDE ( CTest )
ADD_SUBDIRECTORY ( testshade )
# Ugh, I can't seem to ADD_SUBDIRECTORY the testsuite dir if it's not a true
# subdirectory of source, so we fake it by creating a link.  FIXME!
EXECUTE_PROCESS ( COMMAND cmake -E create_symlink ${PROJECT_SOURCE_DIR}/../testsuite ${PROJECT_SOURCE_DIR}/testsuite )
ADD_SUBDIRECTORY ( testsuite )


#########################################################################
# Packaging
SET ( CPACK_PACKAGE_VERSION_MAJOR ${OSL_VERSION_MAJOR} )
SET ( CPACK_PACKAGE_VERSION_MINOR ${OSL_VERSION_MINOR} )
SET ( CPACK_PACKAGE_VERSION_PATCH ${OSL_VERSION_PATCH} )
# "Vendor" is only used in copyright notices, so we use the same thing that
# the rest of the copyright notices say.
SET ( CPACK_PACKAGE_VENDOR "Sony Pictures Imageworks")
SET ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenShadingLanguage is..." )
SET ( CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/doc/Description.txt" )
SET ( CPACK_PACKAGE_FILE_NAME OSL-${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}-${platform} )
#SET ( CPACK_PACKAGE_INSTALL_DIRECTORY ${PROJECT_SOURCE_DIR}/.. )
EXEC_PROGRAM ( "cmake -E copy ${PROJECT_SOURCE_DIR}/../LICENSE ${CMAKE_BINARY_DIR}/License.txt" )
SET ( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/License.txt" )
EXEC_PROGRAM ( "cmake -E copy ${PROJECT_SOURCE_DIR}/../README ${CMAKE_BINARY_DIR}/Readme.txt" )
SET ( CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/Readme.txt" )
SET ( CPACK_RESOURCE_FILE_WELCOME "${PROJECT_SOURCE_DIR}/doc/Welcome.txt" )
#SET ( CPACK_PACKAGE_EXECUTABLES I'm not sure what this is for )
#SET ( CPACK_STRIP_FILES Do we need this? )
IF ( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
    SET ( CPACK_GENERATOR "TGZ;STGZ;RPM;DEB" )
    SET ( CPACK_SOURCE_GENERATOR "TGZ" )
ENDIF ()
IF ( APPLE )
    SET ( CPACK_GENERATOR "TGZ;STGZ;PackageMaker" )
    SET ( CPACK_SOURCE_GENERATOR "TGZ" )
ENDIF ()
SET ( CPACK_SOURCE_PACKAGE_FILE_NAME OSL-${OSL_VERSION_MAJOR}.${OSL_VERSION_MINOR}.${OSL_VERSION_PATCH}-source )
#SET ( CPACK_SOURCE_STRIP_FILES ...FIXME... )
SET ( CPACK_SOURCE_IGNORE_FILES ".*~" )
INCLUDE ( CPack )
