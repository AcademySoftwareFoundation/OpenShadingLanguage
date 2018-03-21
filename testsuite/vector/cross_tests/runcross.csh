#!/bin/csh

echo "\n"
echo "*******************"
echo "osl_cross_w16dvw16dvw16v"
echo "*******************"

set cross_cmd=`testshade --batched -g 200 200 test_cross_w16dvw16dvw16v -od uint8 -o dcross dcross_w16dvw16dvw16v_out.tif -o dxcross dxcross_w16dvw16dvw16v_out.tif -o dycross dycross_w16dvw16dvw16v_out.tif` 

set cross_cmd_ref=`testshade -g 200 200 test_cross_w16dvw16dvw16v -od uint8 -o dcross dcross_w16dvw16dvw16v_ref.tif -o dxcross dxcross_w16dvw16dvw16v_ref.tif -o dycross dycross_w16dvw16dvw16v_ref.tif` 

set cross_dx_idiff=`idiff dxcross_w16dvw16dvw16v_ref.tif dxcross_w16dvw16dvw16v_out.tif`
printf "$cross_dx_idiff \n"

set cross_dy_idiff=`idiff dycross_w16dvw16dvw16v_ref.tif dycross_w16dvw16dvw16v_out.tif`
printf "$cross_dy_idiff \n"



#****************

echo "\n"
echo "*******************"
echo "osl_cross_w16dvw16vw16dv"
echo "*******************"


set cross_cmd=`testshade --batched -g 200 200 test_cross_w16dvw16vw16dv -od uint8 -o dcross dcross_w16dvw16vw16dv_out.tif -o dxcross dxcross_w16dvw16vw16dv_out.tif -o dycross dycross_w16dvw16vw16dv_out.tif`

set cross_cmd_ref=`testshade -g 200 200 test_cross_w16dvw16vw16dv -od uint8 -o dcross dcross_w16dvw16vw16dv_ref.tif -o dxcross dxcross_w16dvw16vw16dv_ref.tif -o dycross dycross_w16dvw16vw16dv_ref.tif`


set cross_dx_idiff=`idiff dxcross_w16dvw16vw16dv_ref.tif dxcross_w16dvw16vw16dv_out.tif`
printf "$cross_dx_idiff \n"

set cross_dy_idiff=`idiff dycross_w16dvw16vw16dv_ref.tif dycross_w16dvw16vw16dv_out.tif`
printf "$cross_dy_idiff \n"

#****************


echo "\n"
echo "*******************"
echo "osl_cross_w16dvw16dvw16dv"
echo "*******************"

set cross_cmd=`testshade --batched -g 200 200 test_cross_w16dvw16dvw16dv -od uint8 -o dcross dcross_w16dvw16dvw16dv_out.tif -o dxcross dxcross_w16dvw16dvw16dv_out.tif -o dycross dycross_w16dvw16dvw16dv_out.tif`

set cross_cmd_ref=`testshade -g 200 200 test_cross_w16dvw16dvw16dv -od uint8 -o dcross dcross_w16dvw16dvw16dv_ref.tif -o dxcross dxcross_w16dvw16dvw16dv_ref.tif -o dycross dycross_w16dvw16dvw16dv_ref.tif`

set cross_idiff=`idiff dcross_w16dvw16dvw16dv_ref.tif dcross_w16dvw16dvw16dv_out.tif`
printf "$cross_idiff \n"

set cross_dx_idiff=`idiff dxcross_w16dvw16dvw16dv_ref.tif dxcross_w16dvw16dvw16dv_out.tif`
printf "$cross_dx_idiff \n"

set cross_dy_idiff=`idiff dycross_w16dvw16dvw16dv_ref.tif dycross_w16dvw16dvw16dv_out.tif`
printf "$cross_dy_idiff \n"



echo "\n"
echo "*******************"
echo "osl_cross_w16vw16vw16v"
echo "*******************"

set cross_cmd=`testshade --batched -g 200 200 test_cross_w16vw16vw16v -o dcross dcross_w16vw16vw16v_out.tif`

set cross_cmd_ref=`testshade -g 200 200 test_cross_w16vw16vw16v  -o dcross dcross_w16vw16vw16v_ref.tif`

set cross_idiff=`idiff dcross_w16vw16vw16v_ref.tif dcross_w16vw16vw16v_out.tif`
printf "$cross_idiff \n"
#testshade --batched -g 200 200 test_cross_w16dvw16dvw16dv -od uint8 -o dcross dcross_ref.tif -o dxcross# dxcross_ref.tif -o dycross dycross_ref.tif
