# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

set ( oslc_srcs oslcmain.cpp )

# don't want to link oslexec but oslcomp uses these symbols
if (NOT BUILD_SHARED_LIBS)
    list (APPEND oslc_srcs
         ../liboslexec/oslexec.cpp
         ../liboslexec/typespec.cpp)
endif ()

add_executable ( oslc ${oslc_srcs} )
target_link_libraries ( oslc PRIVATE oslcomp ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
install_targets (oslc)
