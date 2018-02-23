#!/bin/csh

echo "\n"
echo "*******************"
echo "Catmull-Rom spline..."
echo "*******************"


set ITER_NUM=2
#testshade --batched -g 4 4 -t 1 --iters 1 --stats -o Fspline1 a.tif -o DxFspline1 b.tif test_bezier_df_df_df
#testshade -g 4 4 -t 1 --iters 1 --stats -od uint8 -o Fspline1 a_ref.tif -o DxFspline1 b_ref.tif test_bezier_df_f_df

set cmr_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 catmullrom_out.tif -o DxFspline1 catmullrom_dx_out.tif test_catmullrom_df_f_df| grep "Run  :"`

printf "Batched time: "
printf "$cmr_cmd \n"

set cmr_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 catmullrom_ref.tif -o DxFspline1 catmullrom_dx_ref.tif test_catmullrom_df_f_df | grep "Run  :"`

printf "Non-batched time: "
printf "$cmr_cmd_ref\n"

set cmr_idiff=`idiff catmullrom_ref.tif catmullrom_out.tif`
printf "$cmr_idiff \n"

set cmr_dx_idiff=`idiff catmullrom_dx_ref.tif catmullrom_dx_out.tif`
printf "$cmr_dx_idiff \n"


#if ( $linear_idiff=="PASS") then
#	printf "Catmull-rom PASS"

echo "\n"
echo "*******************"
echo "Bezier spline..."
echo "*******************"

set bez_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 bezier_out.tif -o DxFspline1 bezier_dx_out.tif test_bezier_df_f_df| grep "Run  :"`

printf "Batched time: "
printf "$bez_cmd \n"

set bez_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 bezier_ref.tif -o DxFspline1 bezier_dx_ref.tif test_bezier_df_f_df| grep "Run  :"`

printf "Non-batched time: "
printf "$bez_cmd_ref\n"

set bez_idiff=`idiff bezier_ref.tif bezier_out.tif`
printf "$bez_idiff \n"

set bez_dx_idiff=`idiff bezier_dx_ref.tif bezier_dx_out.tif`
printf "$bez_dx_idiff \n"


echo "\n"
echo "*******************"
echo "Bspline spline..."
echo "*******************"

set bsp_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 bspline_out.tif -o DxFspline1 bspline_dx_out.tif test_bspline_df_f_df| grep "Run  :"`
printf "Batched time: "
printf "$bsp_cmd \n"

set bsp_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 bspline_ref.tif -o DxFspline1 bspline_dx_ref.tif test_bspline_df_f_df| grep "Run  :"`
printf "Non-batched time: "
printf "$bsp_cmd_ref\n"

set bsp_idiff=`idiff bspline_ref.tif bspline_out.tif`
printf "$bsp_idiff \n"

set bsp_dx_idiff=`idiff bspline_dx_ref.tif bspline_dx_out.tif`
printf "$bsp_dx_idiff \n"


echo "\n"
echo "*******************"
echo "Hermite spline..."
echo "*******************"

set her_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 hermite_out.tif -o DxFspline1 hermite_dx_out.tif test_hermite_df_f_df| grep "Run  :"`

printf "Batched time: "
printf "$her_cmd \n"

set her_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 hermite_ref.tif -o DxFspline1 hermite_dx_ref.tif test_hermite_df_f_df| grep "Run  :"`

printf "Non-batched time: "
printf "$her_cmd_ref\n"

set her_idiff=`idiff hermite_ref.tif hermite_out.tif`
printf "$her_idiff \n"

set her_dx_idiff=`idiff hermite_dx_ref.tif hermite_dx_out.tif`
printf "$her_dx_idiff \n"



echo "\n"
echo "*******************"
echo "Linear spline..."
echo "*******************"

set lin_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 linear_out.tif -o DxFspline1 linear_dx_out.tif test_linear_df_f_df| grep "Run  :"`
printf "Batched time: "
printf "$lin_cmd \n"

set lin_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 linear_ref.tif -o DxFspline1 linear_dx_ref.tif test_linear_df_f_df| grep "Run  :"`

printf "Non-batched time: "
printf "$lin_cmd_ref\n"

set lin_idiff=`idiff linear_ref.tif linear_out.tif`
printf "$lin_idiff \n"

set lin_dx_idiff=`idiff linear_dx_ref.tif linear_dx_out.tif`
printf "$lin_dx_idiff \n"



echo "\n"
echo "*******************"
echo "Constant spline..."
echo "*******************"

set con_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 constant_out.tif -o DxFspline1 constant_dx_out.tif test_constant_df_f_df| grep "Run  :"`

printf "Batched time: "
printf "$con_cmd \n"

set con_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 constant_ref.tif -o DxFspline1 constant_dx_ref.tif test_constant_df_f_df| grep "Run  :"`
printf "Non-batched time: "
printf "$con_cmd_ref\n"

set con_idiff=`idiff constant_ref.tif constant_out.tif`
printf "$con_idiff \n"

set con_dx_idiff=`idiff constant_dx_ref.tif constant_dx_out.tif`
printf "$con_dx_idiff \n"



