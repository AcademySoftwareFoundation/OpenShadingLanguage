# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set (CMAKE_AUTOMOC ON)
if (Qt5_POSITION_INDEPENDENT_CODE OR Qt6_POSITION_INDEPENDENT_CODE)
    set (CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
set (osltoy_srcs
     osltoymain.cpp osltoyapp.cpp codeeditor.cpp osltoyrenderer.cpp)
add_executable (osltoy ${osltoy_srcs})
target_include_directories (osltoy BEFORE PRIVATE ${OpenImageIO_INCLUDES})
set_target_properties (osltoy PROPERTIES FOLDER "Tools")
target_link_libraries (osltoy
    PRIVATE
        oslexec oslcomp
        $<TARGET_NAME_IF_EXISTS:Qt5::Core>
        $<TARGET_NAME_IF_EXISTS:Qt5::Gui>
        $<TARGET_NAME_IF_EXISTS:Qt5::Widgets>
        $<TARGET_NAME_IF_EXISTS:Qt6::Core>
        $<TARGET_NAME_IF_EXISTS:Qt6::Gui>
        $<TARGET_NAME_IF_EXISTS:Qt6::Widgets>
        ${CMAKE_DL_LIBS})
if (Qt5_FOUND)
    target_compile_definitions(osltoy PRIVATE OSL_QT_MAJOR=5)
endif ()
if (Qt6_FOUND)
    target_compile_definitions(osltoy PRIVATE OSL_QT_MAJOR=6)
endif ()
install_targets (osltoy)
