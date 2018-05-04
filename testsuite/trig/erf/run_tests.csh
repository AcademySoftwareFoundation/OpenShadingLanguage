#!/bin/csh


oslc test_erf_w16ff.osl

oslc test_erf_w16f_w16f.osl

oslc test_erf_w16dfw16df.osl

#############################
# osl_erf_w16ff
#
#############################

testshade  --batched -g 200 200 test_erf_w16f_f \
                        -od uint8 -o res wff_out.tif -o res_m m_wff_out.tif 
testshade -g 200 200 test_erf_w16f_f \
                      -od uint8 -o res wff_ref.tif -o res_m m_wff_ref.tif
                      
idiff wff_ref.tif wff_out.tif
idiff m_wff_ref.tif m_wff_out.tif;

#############################
# osl_erf_w16f_w16f
#
#############################

testshade --batched -g 200 200 test_erf_w16f_w16f -od uint8 -o res wfwf_out.tif -o res_m m_wfwf_out.tif
testshade -g 200 200 test_erf_w16f_w16f -od uint8 -o res wfwf_ref.tif -o res_m m_wfwf_ref.tif

idiff wfwf_ref.tif wfwf_out.tif
idiff m_wfwf_ref.tif m_wfwf_out.tif


#############################
# osl_erf_w16df_w16df
#
##############################

testshade --batched -g 200 200 test_erf_w16df_w16df -od uint8 -o res wdfwdf_out.tif -o Dxres wdfwdf_dx_out.tif -o res_m m_wdfwdf_out.tif -o Dxres_m m_wdfwdf_dx_out.tif

testshade  -g 200 200 test_erf_w16df_w16df -od uint8 -o res wdfwdf_ref.tif -o Dxres wdfwdf_dx_ref.tif -o res_m m_wdfwdf_ref.tif -o Dxres_m m_wdfwdf_dx_ref.tif

idiff wdfwdf_out.tif wdfwdf_ref.tif
idiff wdfwdf_dx_out.tif  wdfwdf_dx_ref.tif

#Masked
idiff -fail 0.004 m_wdfwdf_out.tif m_wdfwdf_ref.tif
idiff m_wdfwdf_dx_out.tif m_wdfwdf_dx_ref.tif 



