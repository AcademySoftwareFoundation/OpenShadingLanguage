# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

find_package(Threads REQUIRED)

function(ADD_BSDL_LIBRARY NAME)
    cmake_parse_arguments(PARSE_ARGV 1 bsdl "" "SUBDIR" "SPECTRAL_COLOR_SPACES")
    # Bootstrap version of BSDL (without luts)
    add_library(BSDL_BOOTSTRAP INTERFACE)
    target_include_directories(BSDL_BOOTSTRAP INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/${bsdl_SUBDIR}/include)
    target_link_libraries(BSDL_BOOTSTRAP INTERFACE ${ARNOLD_IMATH_TARGETS})

    # LUT generation tool
    set(BSDL_GEN_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${bsdl_SUBDIR}/geninclude)
    add_executable (genluts ${CMAKE_CURRENT_SOURCE_DIR}/${bsdl_SUBDIR}/src/genluts.cpp)
    target_link_libraries(genluts PRIVATE BSDL_BOOTSTRAP Threads::Threads)
    file(MAKE_DIRECTORY ${BSDL_GEN_HEADERS}/BSDL/SPI)
    add_custom_command(TARGET genluts POST_BUILD USES_TERMINAL COMMAND $<TARGET_FILE:genluts> ${BSDL_GEN_HEADERS}/BSDL/SPI
                    COMMENT "Generating BSDL lookup tables ...")

    if (DEFINED bsdl_SPECTRAL_COLOR_SPACES)
        add_executable(jakobhanika_luts ${CMAKE_CURRENT_SOURCE_DIR}/${bsdl_SUBDIR}/src/jakobhanika_luts.cpp)
        target_link_libraries(jakobhanika_luts PRIVATE Threads::Threads)
        foreach(CS ${bsdl_SPECTRAL_COLOR_SPACES})
            set(JACOBHANIKA_${CS} ${CMAKE_CURRENT_BINARY_DIR}/jakobhanika_${CS}.cpp)
            list(APPEND BSDL_LUTS_CPP ${JACOBHANIKA_${CS}})
            add_custom_command(
                OUTPUT ${JACOBHANIKA_${CS}}
                USES_TERMINAL
                COMMAND $<TARGET_FILE:jakobhanika_luts> 64 ${JACOBHANIKA_${CS}} ${CS}
                DEPENDS jakobhanika_luts
                COMMENT "Generating Jakob-Hanika RGB-Spectrum ${CS} tables")
        endforeach()
        set(${NAME}_LUTS_CPP ${BSDL_LUTS_CPP} PARENT_SCOPE)
    endif()

    # Final BSDL library (with luts)
    add_library(${NAME} INTERFACE)
    target_include_directories(${NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/${bsdl_SUBDIR}/include)
    target_link_libraries(${NAME} INTERFACE Imath::Imath)
    target_include_directories(${NAME} INTERFACE ${BSDL_GEN_HEADERS})
    add_dependencies(${NAME} genluts)
endfunction()
