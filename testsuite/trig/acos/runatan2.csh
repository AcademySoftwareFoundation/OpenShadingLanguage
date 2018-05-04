#!/bin/csh

echo "\n"
echo "*******************"
echo "osl_atan2_w16f_w16f_w16f"
echo "*******************"

set atan2_cmd=`testshade --batched  -g 200 200 test_atan2_w16f_w16f_w16f  -od uint8 -o res wfwfwf_out.tif -o res_m m_wfwfwf_out.tif`
set atan2_cmd_ref=`testshade -g 200 200 test_atan2_w16f_w16f_w16f -od uint8 -o res wfwfwf_ref.tif -o res_m m_wfwfwf_ref.tif`

set atan2_idiff=`idiff wfwfwf_ref.tif wfwfwf_out.tif`
printf "$atan2_idiff \n"

set atan2_m_idiff=`idiff m_wfwfwf_ref.tif m_wfwfwf_out.tif`
printf "$atan2_m_idiff \n"




echo "\n"
echo "*******************"
echo "osl_atan2_w16df_w16df_w16df"
echo "*******************"



set atan2_cmd=`testshade --batched  -g 200 200 test_atan2_w16df_w16df_w16df -od uint8 -o res wdfwdfwdf_out.tif \
                                                                                      -o Dxres wdfwdfwdf_dx_out.tif \
                                                                                      -o res_m m_wdfwdfwdf_out.tif\
                                                                                      -o Dxres_m m_wdfwdfwdf_dx_out.tif`
                                                                                      
set atan2_cmd=`testshade  -g 200 200 test_atan2_w16df_w16df_w16df -od uint8 -o res wdfwdfwdf_ref.tif \
                                                                                      -o Dxres wdfwdfwdf_dx_ref.tif \
                                                                                      -o res_m m_wdfwdfwdf_ref.tif\
                                                                                      -o Dxres_m m_wdfwdfwdf_dx_ref.tif`

set atan2_idiff=`idiff wdfwdfwdf_ref.tif wdfwdfwdf_out.tif`
printf "$atan2_idiff \n"

set atan2_idiff_mdx=`idiff wdfwdfwdf_dx_ref.tif wdfwdfwdf_dx_out.tif`
printf "$atan2_idiff_mdx \n"

#Masked tests
set atan2_idiff_m=`idiff m_wdfwdfwdf_ref.tif m_wdfwdfwdf_out.tif`
printf "$atan2_idiff_m \n"

set atan2_idiff_mdx=`idiff m_wdfwdfwdf_dx_ref.tif m_wdfwdfwdf_dx_out.tif`
printf "$atan2_idiff_mdx \n"



echo "\n"
echo "*******************"
echo "osl_atan2_w16df_w16df_w16f"
echo "*******************"



set atan2_cmd=`testshade --batched  -g 200 200 test_atan2_w16df_w16df_w16f -od uint8 -o res wdfwdfwf_out.tif \
                                                                                      -o Dxres wdfwdfwf_dx_out.tif \
                                                                                      -o res_m m_wdfwdfwf_out.tif\
                                                                                      -o Dxres_m m_wdfwdfwf_dx_out.tif`
                                                                                      
set atan2_cmd=`testshade  -g 200 200 test_atan2_w16df_w16df_w16f -od uint8 -o res wdfwdfwf_ref.tif \
                                                                                      -o Dxres wdfwdfwf_dx_ref.tif \
                                                                                      -o res_m m_wdfwdfwf_ref.tif\
                                                                                      -o Dxres_m m_wdfwdfwf_dx_ref.tif`

set atan2_idiff=`idiff wdfwdfwf_ref.tif wdfwdfwf_out.tif`
printf "$atan2_idiff \n"

set atan2_idiff_mdx=`idiff wdfwdfwf_dx_ref.tif wdfwdfwf_dx_out.tif`
printf "$atan2_idiff_mdx \n"

#Masked tests
set atan2_idiff_m=`idiff m_wdfwdfwf_ref.tif m_wdfwdfwf_out.tif`
printf "$atan2_idiff_m \n"

set atan2_idiff_mdx=`idiff m_wdfwdfwf_dx_ref.tif m_wdfwdfwf_dx_out.tif`
printf "$atan2_idiff_mdx \n"


echo "\n"
echo "*******************"
echo "osl_atan2_w16df_w16f_w16df"
echo "*******************"



set atan2_cmd=`testshade --batched  -g 200 200 test_atan2_w16df_w16f_w16df -od uint8 -o res wdfwfwdf_out.tif \
                                                                                      -o Dxres wdfwfwdf_dx_out.tif \
                                                                                      -o res_m m_wdfwfwdf_out.tif\
                                                                                      -o Dxres_m m_wdfwfwdf_dx_out.tif`
                                                                                      
set atan2_cmd=`testshade  -g 200 200 test_atan2_w16df_w16f_w16df -od uint8 -o res wdfwfwdf_ref.tif \
                                                                                      -o Dxres wdfwfwdf_dx_ref.tif \
                                                                                      -o res_m m_wdfwfwdf_ref.tif\
                                                                                      -o Dxres_m m_wdfwfwdf_dx_ref.tif`

set atan2_idiff=`idiff wdfwfwdf_ref.tif wdfwfwdf_out.tif`
printf "$atan2_idiff \n"

set atan2_idiff_mdx=`idiff wdfwfwdf_dx_ref.tif wdfwfwdf_dx_out.tif`
printf "$atan2_idiff_mdx \n"

#Masked tests
set atan2_idiff_m=`idiff m_wdfwfwdf_ref.tif m_wdfwfwdf_out.tif`
printf "$atan2_idiff_m \n"

set atan2_idiff_mdx=`idiff m_wdfwfwdf_dx_ref.tif m_wdfwfwdf_dx_out.tif`
printf "$atan2_idiff_mdx \n"



echo "\n"
echo "*******************"
echo "osl_atan2_w16dv_w16dv_w16dv"
echo "*******************"



set atan2_cmd=`testshade --batched  -g 200 200 test_atan2_w16dv_w16dv_w16dv -od uint8 -o res wdvwdvwdv_out.tif \
                                                                                      -o Dxres wdvwdvwdv_dx_out.tif \
                                                                                      -o res_m m_wdvwdvwdv_out.tif\
                                                                                      -o Dxres_m m_wdvwdvwdv_dx_out.tif`
                                                                                      
set atan2_cmd=`testshade  -g 200 200 test_atan2_w16dv_w16dv_w16dv -od uint8 -o res wdvwdvwdv_ref.tif \
                                                                                      -o Dxres wdvwdvwdv_dx_ref.tif \
                                                                                      -o res_m m_wdvwdvwdv_ref.tif\
                                                                                      -o Dxres_m m_wdvwdvwdv_dx_ref.tif`

set atan2_idiff=`idiff wdvwdvwdv_ref.tif wdvwdvwdv_out.tif`
printf "$atan2_idiff \n"

set atan2_idiff_mdx=`idiff wdvwdvwdv_dx_ref.tif wdvwdvwdv_dx_out.tif`
printf "$atan2_idiff_mdx \n"

#Masked tests
set atan2_idiff_m=`idiff m_wdvwdvwdv_ref.tif m_wdvwdvwdv_out.tif`
printf "$atan2_idiff_m \n"

set atan2_idiff_mdx=`idiff m_wdvwdvwdv_dx_ref.tif m_wdvwdvwdv_dx_out.tif`
printf "$atan2_idiff_mdx \n"


echo "\n"
echo "*******************"
echo "osl_atan2_w16dv_w16dv_w16v"
echo "*******************"



set atan2_cmd=`testshade --batched  -g 200 200 test_atan2_w16dv_w16dv_w16v -od uint8 -o res wdvwdvwv_out.tif \
                                                                                      -o Dxres wdvwdvwv_dx_out.tif \
                                                                                      -o res_m m_wdvwdvwv_out.tif\
                                                                                      -o Dxres_m m_wdvwdvwv_dx_out.tif`
                                                                                      
set atan2_cmd=`testshade  -g 200 200 test_atan2_w16dv_w16dv_w16v -od uint8 -o res wdvwdvwv_ref.tif \
                                                                                      -o Dxres wdvwdvwv_dx_ref.tif \
                                                                                      -o res_m m_wdvwdvwv_ref.tif\
                                                                                      -o Dxres_m m_wdvwdvwv_dx_ref.tif`

set atan2_idiff=`idiff wdvwdvwv_ref.tif wdvwdvwv_out.tif`
printf "$atan2_idiff \n"

set atan2_idiff_mdx=`idiff wdvwdvwv_dx_ref.tif wdvwdvwv_dx_out.tif`
printf "$atan2_idiff_mdx \n"

#Masked tests
set atan2_idiff_m=`idiff m_wdvwdvwv_ref.tif m_wdvwdvwv_out.tif`
printf "$atan2_idiff_m \n"

set atan2_idiff_mdx=`idiff m_wdvwdvwv_dx_ref.tif m_wdvwdvwv_dx_out.tif`
printf "$atan2_idiff_mdx \n"



echo "\n"
echo "*******************"
echo "osl_atan2_w16dv_w16v_w16dv"
echo "*******************"



set atan2_cmd=`testshade --batched  -g 200 200 test_atan2_w16dv_w16v_w16dv -od uint8 -o res wdvwvwdv_out.tif \
                                                                                      -o Dxres wdvwvwdv_dx_out.tif \
                                                                                      -o res_m m_wdvwvwdv_out.tif\
                                                                                      -o Dxres_m m_wdvwvwdv_dx_out.tif`
                                                                                      
set atan2_cmd=`testshade  -g 200 200 test_atan2_w16dv_w16v_w16dv -od uint8 -o res wdvwvwdv_ref.tif \
                                                                                      -o Dxres wdvwvwdv_dx_ref.tif \
                                                                                      -o res_m m_wdvwvwdv_ref.tif\
                                                                                      -o Dxres_m m_wdvwvwdv_dx_ref.tif`

set atan2_idiff=`idiff wdvwvwdv_ref.tif wdvwvwdv_out.tif`
printf "$atan2_idiff \n"

set atan2_idiff_mdx=`idiff wdvwvwdv_dx_ref.tif wdvwvwdv_dx_out.tif`
printf "$atan2_idiff_mdx \n"

#Masked tests
set atan2_idiff_m=`idiff m_wdvwvwdv_ref.tif m_wdvwvwdv_out.tif`
printf "$atan2_idiff_m \n"

set atan2_idiff_mdx=`idiff m_wdvwvwdv_dx_ref.tif m_wdvwvwdv_dx_out.tif`
printf "$atan2_idiff_mdx \n"



echo "\n"
echo "*******************"
echo "osl_atan2_w16v_w16v_w16v"
echo "*******************"

set atan2_cmd=`testshade --batched  -g 200 200 test_atan2_w16v_w16v_w16v  -od uint8 -o res wvwvwv_out.tif -o res_m m_wvwvwv_out.tif`
set atan2_cmd_ref=`testshade -g 200 200 test_atan2_w16v_w16v_w16v -od uint8 -o res wvwvwv_ref.tif -o res_m m_wvwvwv_ref.tif`

set atan2_idiff=`idiff wvwvwv_ref.tif wvwvwv_out.tif`
printf "$atan2_idiff \n"

set atan2_m_idiff=`idiff m_wvwvwv_ref.tif m_wvwvwv_out.tif`
printf "$atan2_m_idiff \n"

