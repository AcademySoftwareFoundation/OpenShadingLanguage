#!/bin/csh

echo "\n"
echo "*******************"
echo "osl_dot_w16dfw16dvw16v"
echo "*******************"

set dot_cmd=`testshade --batched -g 200 200 test_dot_w16dfw16dvw16v -od uint8 -o ddot ddot_w16dfw16dvw16v_out.tif -o dxdot dxdot_w16dfw16dvw16v_out.tif -o dydot dydot_w16dfw16dvw16v_out.tif`
 

set dot_cmd_ref=`testshade  -g 200 200 test_dot_w16dfw16dvw16v -od uint8 -o ddot ddot_w16dfw16dvw16v_ref.tif -o dxdot dxdot_w16dfw16dvw16v_ref.tif -o dydot dydot_w16dfw16dvw16v_ref.tif`


set dot_diff=`idiff ddot_w16dfw16dvw16v_out.tif ddot_w16dfw16dvw16v_ref.tif`
printf "$dot_diff \n"
set dxdot_diff=`idiff dxdot_w16dfw16dvw16v_out.tif dxdot_w16dfw16dvw16v_ref.tif`
printf "$dxdot_diff \n"
set dydot_diff=`idiff dydot_w16dfw16dvw16v_out.tif dydot_w16dfw16dvw16v_ref.tif`
printf "$dydot_diff \n"


#****************

echo "\n"
echo "*******************"
echo "osl_dot_w16dfw16vw16dv"
echo "*******************"

set dot_cmd=`testshade --batched -g 200 200 test_dot_w16dfw16vw16dv -od uint8 -o ddot ddot_w16dfw16vw16dv_out.tif -o dxdot dxdot_w16dfw16vw16dv_out.tif -o dydot dydot_w16dfw16vw16dv_out.tif`
 

set dot_cmd_ref=`testshade  -g 200 200 test_dot_w16dfw16vw16dv -od uint8 -o ddot ddot_w16dfw16vw16dv_ref.tif -o dxdot dxdot_w16dfw16vw16dv_ref.tif -o dydot dydot_w16dfw16vw16dv_ref.tif`


set dot_diff=`idiff ddot_w16dfw16vw16dv_out.tif ddot_w16dfw16vw16dv_ref.tif`
printf "$dot_diff \n"
set dxdot_diff=`idiff dxdot_w16dfw16vw16dv_out.tif dxdot_w16dfw16vw16dv_ref.tif`
printf "$dxdot_diff \n"
set dydot_diff=`idiff dydot_w16dfw16vw16dv_out.tif dydot_w16dfw16vw16dv_ref.tif`
printf "$dydot_diff \n"

#****************

echo "\n"
echo "*******************"
echo "osl_dot_w16dfw16dvw16dv"
echo "*******************"

set dot_cmd=`testshade --batched -g 200 200 test_dot_w16dfw16dvw16dv -od uint8 -o ddot ddot_w16dfw16dvw16dv_out.tif -o dxdot dxdot_w16dfw16dvw16dv_out.tif -o dydot dydot_w16dfw16dvw16dv_out.tif`
 

set dot_cmd_ref=`testshade  -g 200 200 test_dot_w16dfw16dvw16dv -od uint8 -o ddot ddot_w16dfw16dvw16dv_ref.tif -o dxdot dxdot_w16dfw16dvw16dv_ref.tif -o dydot dydot_w16dfw16dvw16dv_ref.tif`


set dot_diff=`idiff ddot_w16dfw16dvw16dv_out.tif ddot_w16dfw16dvw16dv_ref.tif`
printf "$dot_diff \n"
set dxdot_diff=`idiff dxdot_w16dfw16dvw16dv_out.tif dxdot_w16dfw16dvw16dv_ref.tif`
printf "$dxdot_diff \n"
set dydot_diff=`idiff dydot_w16dfw16dvw16dv_out.tif dydot_w16dfw16dvw16dv_ref.tif`
printf "$dydot_diff \n"

echo "\n"
echo "*******************"
echo "osl_dot_w16fw16vw16v"
echo "*******************"

set dot_cmd=`testshade --batched -g 200 200 test_dot_w16fw16vw16v -od uint8 -o ddot ddot_w16fw16vw16v_out.tif`

set dot_cmd_ref=`testshade  -g 200 200 test_dot_w16fw16vw16v -od uint8 -o ddot ddot_w16fw16vw16v_ref.tif` 

set dot_diff=`idiff ddot_w16fw16vw16v_out.tif ddot_w16fw16vw16v_ref.tif`

printf "$dot_diff \n"


