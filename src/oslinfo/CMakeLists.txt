# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set ( oslinfo_srcs oslinfo.cpp )
add_executable ( oslinfo ${oslinfo_srcs} )
target_link_libraries (oslinfo
                       PRIVATE oslquery
                               ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
install_targets (oslinfo)
