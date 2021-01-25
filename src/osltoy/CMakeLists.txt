# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set (CMAKE_AUTOMOC ON)
if (Qt5_POSITION_INDEPENDENT_CODE)
    set (CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
set (osltoy_srcs
     osltoymain.cpp osltoyapp.cpp codeeditor.cpp osltoyrenderer.cpp)
add_executable (osltoy ${osltoy_srcs})
set_target_properties (osltoy PROPERTIES FOLDER "Tools")
target_link_libraries (osltoy
    PRIVATE
        oslexec
        Qt5::Core Qt5::Gui Qt5::Widgets
        ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
install_targets (osltoy)
