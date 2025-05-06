# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

include(bsdl.cmake)

# BSDL has small tables for doing spectral render in sRGB
# but for other color spaces tell this function which ones
# and it will bake Jakob-Hanika coefficient tables.
add_bsdl_library(BSDL) # SPECTRAL_COLOR_SPACES "ACEScg")