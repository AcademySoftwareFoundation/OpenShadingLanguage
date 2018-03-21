#!/bin/csh

echo "\n"
echo "*******************"
echo "osl_distance_w16dfw16dvw16v"
echo "*******************"

set distance_cmd=`testshade --batched -g 200 200 test_distance_w16dfw16dvw16v -od uint8 -o ddistance ddistance_w16dfw16dvw16v_out.tif -o dxdistance dxdistance_w16dfw16dvw16v_out.tif -o dydistance dydistance_w16dfw16dvw16v_out.tif`
 

set distance_cmd_ref=`testshade  -g 200 200 test_distance_w16dfw16dvw16v -od uint8 -o ddistance ddistance_w16dfw16dvw16v_ref.tif -o dxdistance dxdistance_w16dfw16dvw16v_ref.tif -o dydistance dydistance_w16dfw16dvw16v_ref.tif`


set distance_diff=`idiff ddistance_w16dfw16dvw16v_out.tif ddistance_w16dfw16dvw16v_ref.tif`
printf "$distance_diff \n"
set dxdistance_diff=`idiff dxdistance_w16dfw16dvw16v_out.tif dxdistance_w16dfw16dvw16v_ref.tif`
printf "$dxdistance_diff \n"
set dydistance_diff=`idiff dydistance_w16dfw16dvw16v_out.tif dydistance_w16dfw16dvw16v_ref.tif`
printf "$dydistance_diff \n"


#****************

echo "\n"
echo "*******************"
echo "osl_distance_w16dfw16vw16dv"
echo "*******************"

set distance_cmd=`testshade --batched -g 200 200 test_distance_w16dfw16vw16dv -od uint8 -o ddistance ddistance_w16dfw16vw16dv_out.tif -o dxdistance dxdistance_w16dfw16vw16dv_out.tif -o dydistance dydistance_w16dfw16vw16dv_out.tif`
 

set distance_cmd_ref=`testshade  -g 200 200 test_distance_w16dfw16vw16dv -od uint8 -o ddistance ddistance_w16dfw16vw16dv_ref.tif -o dxdistance dxdistance_w16dfw16vw16dv_ref.tif -o dydistance dydistance_w16dfw16vw16dv_ref.tif`


set distance_diff=`idiff ddistance_w16dfw16vw16dv_out.tif ddistance_w16dfw16vw16dv_ref.tif`
printf "$distance_diff \n"
set dxdistance_diff=`idiff dxdistance_w16dfw16vw16dv_out.tif dxdistance_w16dfw16vw16dv_ref.tif`
printf "$dxdistance_diff \n"
set dydistance_diff=`idiff dydistance_w16dfw16vw16dv_out.tif dydistance_w16dfw16vw16dv_ref.tif`
printf "$dydistance_diff \n"

#****************

echo "\n"
echo "*******************"
echo "osl_distance_w16dfw16dvw16dv"
echo "*******************"

set distance_cmd=`testshade --batched -g 200 200 test_distance_w16dfw16dvw16dv -od uint8 -o ddistance ddistance_w16dfw16dvw16dv_out.tif -o dxdistance dxdistance_w16dfw16dvw16dv_out.tif -o dydistance dydistance_w16dfw16dvw16dv_out.tif`
 

set distance_cmd_ref=`testshade  -g 200 200 test_distance_w16dfw16dvw16dv -od uint8 -o ddistance ddistance_w16dfw16dvw16dv_ref.tif -o dxdistance dxdistance_w16dfw16dvw16dv_ref.tif -o dydistance dydistance_w16dfw16dvw16dv_ref.tif`


set distance_diff=`idiff ddistance_w16dfw16dvw16dv_out.tif ddistance_w16dfw16dvw16dv_ref.tif`
printf "$distance_diff \n"
set dxdistance_diff=`idiff dxdistance_w16dfw16dvw16dv_out.tif dxdistance_w16dfw16dvw16dv_ref.tif`
printf "$dxdistance_diff \n"
set dydistance_diff=`idiff dydistance_w16dfw16dvw16dv_out.tif dydistance_w16dfw16dvw16dv_ref.tif`
printf "$dydistance_diff \n"

echo "\n"
echo "*******************"
echo "osl_distance_w16fw16vw16v"
echo "*******************"

set distance_cmd=`testshade --batched -g 200 200 test_distance_w16fw16vw16v -od uint8 -o ddistance ddistance_w16fw16vw16v_out.tif`

set distance_cmd_ref=`testshade  -g 200 200 test_distance_w16fw16vw16v -od uint8 -o ddistance ddistance_w16fw16vw16v_ref.tif` 

set distance_diff=`idiff ddistance_w16fw16vw16v_out.tif ddistance_w16fw16vw16v_ref.tif`

printf "$distance_diff \n"


