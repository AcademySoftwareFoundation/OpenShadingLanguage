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
        ${OPENIMAGEIO_INCLUDES}
        ${ILMBASE_INCLUDES}
        ${IMATH_INCLUDES}
    )
target_link_libraries (${local_lib}
    PUBLIC
        OpenImageIO::OpenImageIO
        # For OpenEXR/Imath 3.x:
        $<$<TARGET_EXISTS:Imath::Imath>:Imath::Imath>
        $<$<TARGET_EXISTS:Imath::Half>:Imath::Half>
        # For OpenEXR >= 2.4/2.5 with reliable exported targets
        $<$<TARGET_EXISTS:IlmBase::Imath>:IlmBase::Imath>
        $<$<TARGET_EXISTS:IlmBase::Half>:IlmBase::Half>
        $<$<TARGET_EXISTS:IlmBase::IlmThread>:IlmBase::IlmThread>
        $<$<TARGET_EXISTS:IlmBase::Iex>:IlmBase::Iex>
        # For OpenEXR <= 2.3:
        ${ILMBASE_LIBRARIES}
    PRIVATE
        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
        ${CLANG_LIBRARIES} ${LLVM_LIBRARIES} ${LLVM_LDFLAGS}
        ${LLVM_SYSTEM_LIBRARIES})

# link with (system) library to prevent missing symbols inside clangDriver.lib
if (MSVC)
    target_link_libraries (${local_lib} PRIVATE "Version.lib")
endif()

set_target_properties (${local_lib}
    PROPERTIES
        VERSION     ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
        SOVERSION   ${SOVERSION}
        OUTPUT_NAME oslcomp${OSL_LIBNAME_SUFFIX}
        POSITION_INDEPENDENT_CODE ON
    )


if (VISIBILITY_COMMAND)
    set_property (TARGET ${local_lib}
                  APPEND PROPERTY LINK_FLAGS ${VISIBILITY_MAP_COMMAND})
endif ()

install_targets (${local_lib})

