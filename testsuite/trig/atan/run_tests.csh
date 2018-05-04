#!/bin/csh

oslc test_atan_w16ff.osl
oslc test_atan_w16dvw16dv.osl
oslc test_atan_w16vw16v.osl
oslc test_atan_w16dfw16df.osl
oslc test_atan_w16f_w16f.osl

#############################
# osl_atan_w16ff
#
#############################

testshade  --batched -g 200 200 test_atan_w16f_f \
                        -od uint8 -o res wff_out.tif -o res_m m_wff_out.tif 
testshade -g 200 200 test_atan_w16f_f \
                      -od uint8 -o res wff_ref.tif -o res_m m_wff_ref.tif
                      
idiff wff_ref.tif wff_out.tif
idiff m_wff_ref.tif m_wff_out.tif;

#############################
# osl_atan_w16f_w16f
#
#############################

testshade --batched -g 200 200 test_atan_w16f_w16f -od uint8 -o res wfwf_out.tif -o res_m m_wfwf_out.tif
testshade -g 200 200 test_atan_w16f_w16f -od uint8 -o res wfwf_ref.tif -o res_m m_wfwf_ref.tif

idiff wfwf_ref.tif wfwf_out.tif
idiff m_wfwf_ref.tif m_wfwf_out.tif

#############################
# osl_atan_w16df_w16df
#
##############################

testshade --batched -g 200 200 test_atan_w16df_w16df -od uint8 -o res wdfwdf_out.tif -o Dxres wdfwdf_dx_out.tif -o res_m m_wdfwdf_out.tif -o Dxres_m m_wdfwdf_dx_out.tif

testshade  -g 200 200 test_atan_w16df_w16df -od uint8 -o res wdfwdf_ref.tif -o Dxres wdfwdf_dx_ref.tif -o res_m m_wdfwdf_ref.tif -o Dxres_m m_wdfwdf_dx_ref.tif

idiff wdfwdf_out.tif wdfwdf_ref.tif
idiff wdfwdf_dx_out.tif  wdfwdf_dx_ref.tif

#Masked
idiff m_wdfwdf_out.tif m_wdfwdf_ref.tif
idiff m_wdfwdf_dx_out.tif m_wdfwdf_dx_ref.tif 

#############################
# osl_atan_w16v_w16v
#
###############################


testshade --batched -g 200 200 test_atan_w16v_w16v -od uint8 -o res wvwv_out.tif -o res_m m_wvwv_out.tif 

testshade -g 200 200 test_atan_w16v_w16v -od uint8 -o res wvwv_ref.tif -o res_m m_wvwv_ref.tif

idiff wvwv_ref.tif wvwv_out.tif
idiff m_wvwv_ref.tif m_wvwv_out.tif

#############################
# osl_atan_w16dv_w16dv
#
################################


testshade --batched -g 200 200 test_atan_w16dv_w16dv -od uint8 -o res wdvwdv_out.tif -o Dxres wdvwdv_dx_out.tif -o res_m m_wdvwdv_out.tif -o Dxres_m m_wdvwdv_dx_out.tif

testshade  -g 200 200 test_atan_w16dv_w16dv -od uint8 -o res wdvwdv_ref.tif -o Dxres wdvwdv_dx_ref.tif -o res_m m_wdvwdv_ref.tif -o Dxres_m m_wdvwdv_dx_ref.tif

 
idiff wdvwdv_out.tif wdvwdv_ref.tif
idiff wdvwdv_dx_out.tif  wdvwdv_dx_ref.tif

#Masked
idiff m_wdvwdv_out.tif m_wdvwdv_ref.tif
idiff m_wdvwdv_dx_out.tif m_wdvwdv_dx_ref.tif
