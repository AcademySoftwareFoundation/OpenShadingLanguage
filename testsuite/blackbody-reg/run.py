#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


command += testshade("-g 200 200 blackbody_v_temperature -od uint8 -o Cout v_blackbody.tif -o mCout m_v_blackbody.tif")
outputs.append ("v_blackbody.tif")
outputs.append ("m_v_blackbody.tif")

command += testshade("-g 200 200 blackbody_u_temperature -od uint8 -o Cout u_blackbody.tif -o mCout m_u_blackbody.tif")
outputs.append ("u_blackbody.tif")
outputs.append ("m_u_blackbody.tif")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

