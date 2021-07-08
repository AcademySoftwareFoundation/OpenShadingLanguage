#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade("--vary_udxdy --vary_vdxdy --vary_pdxdy -g 32 32 test_filterwidth_u_float -od uint8 -o op fw_u_float.tif "\
                     "-o mop mfw_u_float.tif")
command += testshade("--vary_udxdy --vary_vdxdy --vary_pdxdy -g 32 32 test_filterwidth_u_point -od uint8 -o op fw_u_point.tif")
command += testshade("--vary_udxdy --vary_vdxdy --vary_pdxdy -g 32 32 test_filterwidth_u_vector -od uint8 -o op fw_u_vector.tif")
outputs.append ("fw_u_float.tif")
outputs.append ("mfw_u_float.tif")
outputs.append ("fw_u_point.tif")
outputs.append ("fw_u_vector.tif")

command += testshade("--vary_udxdy --vary_vdxdy --vary_pdxdy -g 32 32 test_filterwidth_v_float -od uint8 -o op fw_v_float.tif")
command += testshade("--vary_udxdy --vary_vdxdy --vary_pdxdy -g 32 32 test_filterwidth_v_point -od uint8 -o op fw_v_point.tif")
command += testshade("--vary_udxdy --vary_vdxdy --vary_pdxdy -g 32 32 test_filterwidth_v_vector -od uint8 -o op fw_v_vector.tif")
outputs.append ("fw_v_float.tif")
outputs.append ("fw_v_point.tif")
outputs.append ("fw_v_vector.tif")

command += testshade("--vary_udxdy --vary_vdxdy --vary_pdxdy -g 32 32 test_filterwidth_v_dvector -od uint8 -o op fw_v_dvector.tif "\
                     "-o dxop dx_fw_v_dvector.tif -o dyop dy_fw_v_dvector.tif "\
                     "-o mop m_fw_v_dvector.tif "\
                     "-o mdxop dx_m_fw_v_dvector.tif -o mdyop dy_m_fw_v_dvector.tif")
outputs.append ("fw_v_dvector.tif")
outputs.append ("dx_fw_v_dvector.tif")
outputs.append ("dy_fw_v_dvector.tif")
outputs.append ("m_fw_v_dvector.tif")
outputs.append ("dx_m_fw_v_dvector.tif")
outputs.append ("dy_m_fw_v_dvector.tif")
                     
command += testshade("--vary_udxdy --vary_vdxdy --vary_pdxdy -g 32 32 test_filterwidth_v_dfloat -od uint8 -o op fw_v_dfloat.tif "\
                     "-o dxop dx_fw_v_dfloat.tif -o dyop dy_fw_v_dfloat.tif "\
                     "-o mop m_fw_v_dfloat.tif "\
                     "-o mdxop dx_m_fw_v_dfloat.tif -o mdyop dy_m_fw_v_dfloat.tif")

outputs.append ("fw_v_dfloat.tif")
outputs.append ("dx_fw_v_dfloat.tif")
outputs.append ("dy_fw_v_dfloat.tif")
outputs.append ("m_fw_v_dfloat.tif")
outputs.append ("dx_m_fw_v_dfloat.tif")
outputs.append ("dy_m_fw_v_dfloat.tif")


# expect a few LSB failures
failthresh = 0.008
failpercent = 3

