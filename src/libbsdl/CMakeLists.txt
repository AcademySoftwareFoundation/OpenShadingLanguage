# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

include(bsdl.cmake)

# BSDL has small tables for doing spectral render in sRGB
# but for other color spaces tell this function which ones
# and it will bake Jakob-Hanika coefficient tables.
add_bsdl_library(BSDL) # SPECTRAL_COLOR_SPACES "ACEScg")


if (OSL_BUILD_TESTS AND BUILD_TESTING)
    add_executable (bsdl_test bsdl_test.cpp)
    set_target_properties (bsdl_test PROPERTIES FOLDER "Unit Tests")
    target_include_directories (bsdl_test BEFORE PRIVATE ${OpenImageIO_INCLUDES})
    target_link_libraries (bsdl_test PRIVATE BSDL OpenImageIO::OpenImageIO)
    add_test (unit_bsdl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bsdl_test)
endif()
