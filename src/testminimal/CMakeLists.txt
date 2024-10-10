# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# The 'testminimal' executable
set ( testminimal_srcs
      testminimal.cpp
      oslmaterial.cpp )

set(include_dirs ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND include_dirs ${CMAKE_SOURCE_DIR}/src/include)
list(APPEND include_dirs ${CMAKE_BINARY_DIR}/include)
list(APPEND include_dirs ${IMATH_INCLUDES})
list(APPEND include_dirs ${OPENEXR_INCLUDES})
list(APPEND include_dirs ${OpenImageIO_INCLUDES})

set ( rs_srcs
    oslmaterial.cpp )

EMBED_LLVM_BITCODE_IN_CPP ( "${rs_srcs}" "_host" "testminimal_llvm_compiled_rs" testminimal_srcs "-DOSL_HOST_RS_BITCODE=1" "${include_dirs}")

add_executable ( testminimal ${testminimal_srcs} )

target_link_libraries (testminimal
                       PRIVATE
                           oslexec oslquery)

install (TARGETS testminimal RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )

osl_optix_target(testminimal)
