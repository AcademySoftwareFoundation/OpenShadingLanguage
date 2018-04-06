#!/bin/csh

oslc test_pow_w16dfw16dfw16df.osl  
oslc test_pow_w16dvw16dvw16v.osl
oslc test_pow_w16dfw16dfw16f.osl   
oslc test_pow_w16dvw16vw16dv.osl
oslc test_pow_w16dfw16fw16df.osl   
oslc test_pow_w16fw16fw16f.osl
oslc test_pow_w16dvw16dvw16dv.osl  
oslc test_pow_w16vw16vw16v.osl


echo "\n"
echo "*******************"
echo "osl_pow_w16f_w16f_w16f"
echo "*******************"

testshade --batched  -g 200 200 test_pow_w16f_w16f_w16f  -od uint8 -o res wfwfwf_out.tif -o res_m m_wfwfwf_out.tif
testshade -g 200 200 test_pow_w16f_w16f_w16f -od uint8 -o res wfwfwf_ref.tif -o res_m m_wfwfwf_ref.tif

idiff wfwfwf_ref.tif wfwfwf_out.tif
idiff m_wfwfwf_ref.tif m_wfwfwf_out.tif





echo "\n"
echo "*******************"
echo "osl_pow_w16df_w16df_w16df"
echo "*******************"

testshade --batched  -g 200 200 test_pow_w16df_w16df_w16df -od uint8 -o res wdfwdfwdf_out.tif \
                                                                                      -o Dxres wdfwdfwdf_dx_out.tif \
                                                                                      -o res_m m_wdfwdfwdf_out.tif\
                                                                                      -o Dxres_m m_wdfwdfwdf_dx_out.tif
                                                                                      
testshade  -g 200 200 test_pow_w16df_w16df_w16df -od uint8 -o res wdfwdfwdf_ref.tif \
                                                                                      -o Dxres wdfwdfwdf_dx_ref.tif \
                                                                                      -o res_m m_wdfwdfwdf_ref.tif\
                                                                                      -o Dxres_m m_wdfwdfwdf_dx_ref.tif

idiff -fail 0.004 wdfwdfwdf_ref.tif wdfwdfwdf_out.tif
idiff -fail 0.004 wdfwdfwdf_dx_ref.tif wdfwdfwdf_dx_out.tif


#Masked tests
idiff -fail 0.004 m_wdfwdfwdf_ref.tif m_wdfwdfwdf_out.tif
idiff -fail 0.004 m_wdfwdfwdf_dx_ref.tif m_wdfwdfwdf_dx_out.tif

echo "\n"
echo "*******************"
echo "osl_pow_w16df_w16df_w16f"
echo "*******************"



testshade --batched  -g 200 200 test_pow_w16df_w16df_w16f -od uint8 -o res wdfwdfwf_out.tif \
                                                                                      -o Dxres wdfwdfwf_dx_out.tif \
                                                                                      -o res_m m_wdfwdfwf_out.tif\
                                                                                      -o Dxres_m m_wdfwdfwf_dx_out.tif
                                                                                      
testshade  -g 200 200 test_pow_w16df_w16df_w16f -od uint8 -o res wdfwdfwf_ref.tif \
                                                                                      -o Dxres wdfwdfwf_dx_ref.tif \
                                                                                      -o res_m m_wdfwdfwf_ref.tif\
                                                                                      -o Dxres_m m_wdfwdfwf_dx_ref.tif

idiff wdfwdfwf_ref.tif wdfwdfwf_out.tif
idiff wdfwdfwf_dx_ref.tif wdfwdfwf_dx_out.tif


#Masked tests
idiff m_wdfwdfwf_ref.tif m_wdfwdfwf_out.tif
idiff m_wdfwdfwf_dx_ref.tif m_wdfwdfwf_dx_out.tif



echo "\n"
echo "*******************"
echo "osl_pow_w16df_w16f_w16df"
echo "*******************"



testshade --batched  -g 200 200 test_pow_w16df_w16f_w16df -od uint8 -o res wdfwfwdf_out.tif \
                                                                                      -o Dxres wdfwfwdf_dx_out.tif \
                                                                                      -o res_m m_wdfwfwdf_out.tif\
                                                                                      -o Dxres_m m_wdfwfwdf_dx_out.tif
                                                                                      
testshade  -g 200 200 test_pow_w16df_w16f_w16df -od uint8 -o res wdfwfwdf_ref.tif \
                                                                                      -o Dxres wdfwfwdf_dx_ref.tif \
                                                                                      -o res_m m_wdfwfwdf_ref.tif\
                                                                                      -o Dxres_m m_wdfwfwdf_dx_ref.tif

idiff wdfwfwdf_ref.tif wdfwfwdf_out.tif
idiff wdfwfwdf_dx_ref.tif wdfwfwdf_dx_out.tif


#Masked tests
idiff m_wdfwfwdf_ref.tif m_wdfwfwdf_out.tif
idiff m_wdfwfwdf_dx_ref.tif m_wdfwfwdf_dx_out.tif




echo "\n"
echo "*******************"
echo "osl_pow_w16dv_w16dv_w16dv"
echo "*******************"



testshade --batched  -g 200 200 test_pow_w16dv_w16dv_w16dv -od uint8 -o res wdvwdvwdv_out.tif \
                                                                                      -o Dxres wdvwdvwdv_dx_out.tif \
                                                                                      -o res_m m_wdvwdvwdv_out.tif\
                                                                                      -o Dxres_m m_wdvwdvwdv_dx_out.tif
                                                                                      
testshade  -g 200 200 test_pow_w16dv_w16dv_w16dv -od uint8 -o res wdvwdvwdv_ref.tif \
                                                                                      -o Dxres wdvwdvwdv_dx_ref.tif \
                                                                                      -o res_m m_wdvwdvwdv_ref.tif\
                                                                                      -o Dxres_m m_wdvwdvwdv_dx_ref.tif

idiff -fail 0.004 wdvwdvwdv_ref.tif wdvwdvwdv_out.tif
idiff -fail 0.004  wdvwdvwdv_dx_ref.tif wdvwdvwdv_dx_out.tif


#Masked tests
idiff -fail 0.004 m_wdvwdvwdv_ref.tif m_wdvwdvwdv_out.tif
idiff -fail 0.004 m_wdvwdvwdv_dx_ref.tif m_wdvwdvwdv_dx_out.tif




echo "\n"
echo "*******************"
echo "osl_pow_w16dv_w16dv_w16v"
echo "*******************"



testshade --batched  -g 200 200 test_pow_w16dv_w16dv_w16v -od uint8 -o res wdvwdvwv_out.tif \
                                                                                      -o Dxres wdvwdvwv_dx_out.tif \
                                                                                      -o res_m m_wdvwdvwv_out.tif\
                                                                                      -o Dxres_m m_wdvwdvwv_dx_out.tif
                                                                                      
testshade  -g 200 200 test_pow_w16dv_w16dv_w16v -od uint8 -o res wdvwdvwv_ref.tif \
                                                                                      -o Dxres wdvwdvwv_dx_ref.tif \
                                                                                      -o res_m m_wdvwdvwv_ref.tif\
                                                                                      -o Dxres_m m_wdvwdvwv_dx_ref.tif

idiff wdvwdvwv_ref.tif wdvwdvwv_out.tif
idiff wdvwdvwv_dx_ref.tif wdvwdvwv_dx_out.tif


#Masked tests
idiff m_wdvwdvwv_ref.tif m_wdvwdvwv_out.tif
idiff m_wdvwdvwv_dx_ref.tif m_wdvwdvwv_dx_out.tif




echo "\n"
echo "*******************"
echo "osl_pow_w16dv_w16v_w16dv"
echo "*******************"



testshade --batched  -g 200 200 test_pow_w16dv_w16v_w16dv -od uint8 -o res wdvwvwdv_out.tif \
                                                                                      -o Dxres wdvwvwdv_dx_out.tif \
                                                                                      -o res_m m_wdvwvwdv_out.tif\
                                                                                      -o Dxres_m m_wdvwvwdv_dx_out.tif
                                                                                      
testshade  -g 200 200 test_pow_w16dv_w16v_w16dv -od uint8 -o res wdvwvwdv_ref.tif \
                                                                                      -o Dxres wdvwvwdv_dx_ref.tif \
                                                                                      -o res_m m_wdvwvwdv_ref.tif\
                                                                                      -o Dxres_m m_wdvwvwdv_dx_ref.tif

idiff wdvwvwdv_ref.tif wdvwvwdv_out.tif
idiff wdvwvwdv_dx_ref.tif wdvwvwdv_dx_out.tif


#Masked tests
idiff m_wdvwvwdv_ref.tif m_wdvwvwdv_out.tif
idiff m_wdvwvwdv_dx_ref.tif m_wdvwvwdv_dx_out.tif




echo "\n"
echo "*******************"
echo "osl_pow_w16v_w16v_w16v"
echo "*******************"

testshade --batched  -g 200 200 test_pow_w16v_w16v_w16v  -od uint8 -o res wvwvwv_out.tif -o res_m m_wvwvwv_out.tif
testshade -g 200 200 test_pow_w16v_w16v_w16v -od uint8 -o res wvwvwv_ref.tif -o res_m m_wvwvwv_ref.tif

idiff wvwvwv_ref.tif wvwvwv_out.tif
idiff m_wvwvwv_ref.tif m_wvwvwv_out.tif


