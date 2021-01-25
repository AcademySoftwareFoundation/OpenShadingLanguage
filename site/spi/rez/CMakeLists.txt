# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

cmake_minimum_required(VERSION 2.8)

project(OSL)

include(RezBuild)

rez_install_dirs(
    include lib share
    # python
    DESTINATION .
)

# Bin specified separately so it can be marked as EXECUTABLE, otherwise it
# installs with only read permissions.
rez_install_dirs(
    bin
    DESTINATION .
    EXECUTABLE
)

rez_install_cmake(
    DESTINATION .
    INCLUDE_DIRS include
    LIBRARY_DIRS lib
    LIBRARIES oslexec oslcomp oslquery
              oslexec_d oslcomp_d oslquery_d
)
