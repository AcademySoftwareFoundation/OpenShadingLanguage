#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


# This test requires an OCIO config. We have one in testsuite/common.
os.environ['OCIO'] = '../common/OpenColorIO/nuke-default/config.ocio'

command = testshade("test")
