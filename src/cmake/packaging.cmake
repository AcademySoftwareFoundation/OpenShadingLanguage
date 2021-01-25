# Copyright Contributors to the Open Shading Languge project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage/

#########################################################################
# Packaging
set (CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
# "Vendor" is only used in copyright notices, so we use the same thing that
# the rest of the copyright notices say.
set (CPACK_PACKAGE_VENDOR ${PROJECT_AUTHORS})
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open Shading Language is the de facto standard for shading in modern path tracers used in film VFX and animation")
set (CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/src/doc/Description.txt")
set (CPACK_PACKAGE_FILE_NAME ${PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-${platform})
file (COPY "${PROJECT_SOURCE_DIR}/LICENSE.md" DESTINATION "${CMAKE_BINARY_DIR}")
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/LICENSE.md")
file (COPY "${PROJECT_SOURCE_DIR}/README.md" DESTINATION "${CMAKE_BINARY_DIR}")
set (CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/README.md")
set (CPACK_RESOURCE_FILE_WELCOME "${PROJECT_SOURCE_DIR}/src/doc/Welcome.txt")
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
set (CPACK_SOURCE_PACKAGE_FILE_NAME ${PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-source)
#set (CPACK_SOURCE_STRIP_FILES ...FIXME...)
set (CPACK_SOURCE_IGNORE_FILES ".*~")
include (CPack)
