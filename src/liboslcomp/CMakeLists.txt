# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set (local_lib oslcomp)
file (GLOB lib_src "*.cpp")
file (GLOB compiler_headers "*.h")

# oslexec symbols used in oslcomp
if (BUILD_SHARED_LIBS)
    list(APPEND lib_src
        ../liboslexec/oslexec.cpp
        ../liboslexec/typespec.cpp
        )
endif ()

FLEX_BISON (osllex.l oslgram.y osl lib_src compiler_headers)

add_library (${local_lib} ${lib_src})
target_include_directories(${local_lib}
    PUBLIC
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
    )
target_link_libraries (${local_lib}
    PUBLIC
        OpenImageIO::OpenImageIO
        $<TARGET_NAME_IF_EXISTS:Imath::Imath>
        $<TARGET_NAME_IF_EXISTS:Imath::Half>
    PRIVATE
        ${CMAKE_DL_LIBS}
        ${CLANG_LIBRARIES} ${LLVM_LIBRARIES} ${LLVM_LDFLAGS}
        ${LLVM_SYSTEM_LIBRARIES})

# link with (system) library to prevent missing symbols inside clangDriver.lib
if (MSVC)
    target_link_libraries (${local_lib} PRIVATE "Version.lib")
endif()

set (link_flags "${VISIBILITY_MAP_COMMAND} ${EXTRA_DSO_LINK_ARGS}")
if (UNIX AND NOT APPLE)
    # Hide symbols from any static dependent libraries embedded here.
    set (link_flags "${link_flags} -Wl,--exclude-libs,ALL")
endif ()

target_compile_features (${local_lib}
                         INTERFACE cxx_std_${DOWNSTREAM_CXX_STANDARD})

set_target_properties (${local_lib}
    PROPERTIES
        VERSION     ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
        SOVERSION   ${SOVERSION}
        OUTPUT_NAME ${local_lib}${OSL_LIBNAME_SUFFIX}
        LINK_FLAGS  ${link_flags}
        POSITION_INDEPENDENT_CODE ON
    )

install_targets (${local_lib})

