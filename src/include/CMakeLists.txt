# Copyright 2009-present Sony Pictures Imageworks, et al.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE

file (GLOB public_headers OSL/*.h)


option (USE_GENERATED_EXPORT_HEADER "Use CMake-generated export header" OFF)
if (USE_GENERATED_EXPORT_HEADER)
    include (GenerateExportHeader)
    set (exportextras "\n#define OSL_EXPORT OSL_API\n")
    generate_export_header (OSL
          # [BASE_NAME baseName]
          EXPORT_FILE_NAME export.h
          EXPORT_MACRO_NAME OSL_API
          NO_EXPORT_MACRO_NAME OSL_LOCAL
          STATIC_DEFINE OSL_STATIC_DEFINE
          DEPRECATED_MACRO_NAME OSL_DEPRECATED
          # [NO_DEPRECATED_MACRO_NAME noDeprecatedMacroName]
          # [DEFINE_NO_DEPRECATED]
          # [PREFIX_NAME prefix]
          CUSTOM_CONTENT_FROM_VARIABLE exportextras
          )
    list (REMOVE_ITEM public_headers OSL/export.h)
    list (APPEND public_headers "${CMAKE_BINARY_DIR}/src/include/export.h")
endif()


set (versionfile oslversion.h)
if (VERBOSE)
    message (STATUS "Create ${versionfile} from ${versionfile}.in")
endif ()
configure_file (${PROJECT_NAME}/${versionfile}.in "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/${versionfile}" @ONLY)
list (APPEND public_headers "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/${versionfile}")


install (FILES ${public_headers}
         DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
         COMPONENT developer)

set (PROJECT_PUBLIC_HEADERS ${public_headers} PARENT_SCOPE)
