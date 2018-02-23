#!/bin/csh

#These tests call osl_spline_w16fw16ff
echo "\n"
echo "*******************"
echo "Catmull-Rom spline..."
echo "*******************"


set ITER_NUM=2
#testshade --batched -g 4 4 -t 1 --iters 1 --stats -o Fspline1 a.tif -o DxFspline1 b.tif test_bezier_df_df_df

set cmr_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 catmullrom_out.tif test_catmullrom_f_f_f| grep "Run  :"`

printf "Batched time: "
printf "$cmr_cmd \n"

set cmr_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 catmullrom_ref.tif  test_catmullrom_f_f_f | grep "Run  :"`

printf "Non-batched time: "
printf "$cmr_cmd_ref\n"

set cmr_idiff=`idiff catmullrom_ref.tif catmullrom_out.tif`
printf "$cmr_idiff \n"


echo "\n"
echo "*******************"
echo "Bezier spline..."
echo "*******************"

set bez_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 bezier_out.tif  test_bezier_f_f_f| grep "Run  :"`

printf "Batched time: "
printf "$bez_cmd \n"

set bez_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 bezier_ref.tif test_bezier_f_f_f| grep "Run  :"`

printf "Non-batched time: "
printf "$bez_cmd_ref\n"

set bez_idiff=`idiff bezier_ref.tif bezier_out.tif`
printf "$bez_idiff \n"


echo "\n"
echo "*******************"
echo "Bspline spline..."
echo "*******************"

set bsp_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 bspline_out.tif  test_bspline_f_f_f| grep "Run  :"`
printf "Batched time: "
printf "$bsp_cmd \n"

set bsp_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 bspline_ref.tif  test_bspline_f_f_f| grep "Run  :"`
printf "Non-batched time: "
printf "$bsp_cmd_ref\n"

set bsp_idiff=`idiff bspline_ref.tif bspline_out.tif`
printf "$bsp_idiff \n"




echo "\n"
echo "*******************"
echo "Hermite spline..."
echo "*******************"

set her_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 hermite_out.tif test_hermite_f_f_f| grep "Run  :"`

printf "Batched time: "
printf "$her_cmd \n"

set her_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 hermite_ref.tif  test_hermite_f_f_f| grep "Run  :"`

printf "Non-batched time: "
printf "$her_cmd_ref\n"

set her_idiff=`idiff hermite_ref.tif hermite_out.tif`
printf "$her_idiff \n"



echo "\n"
echo "*******************"
echo "Linear spline..."
echo "*******************"

set lin_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 linear_out.tif test_linear_f_f_f| grep "Run  :"`
printf "Batched time: "
printf "$lin_cmd \n"

set lin_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 linear_ref.tif test_linear_f_f_f| grep "Run  :"`

printf "Non-batched time: "
printf "$lin_cmd_ref\n"

set lin_idiff=`idiff linear_ref.tif linear_out.tif`
printf "$lin_idiff \n"




echo "\n"
echo "*******************"
echo "Constant spline..."
echo "*******************"

set con_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 constant_out.tif test_constant_f_f_f| grep "Run  :"`

printf "Batched time: "
printf "$con_cmd \n"

set con_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline1 constant_ref.tif  test_constant_f_f_f| grep "Run  :"`
printf "Non-batched time: "
printf "$con_cmd_ref\n"

set con_idiff=`idiff constant_ref.tif constant_out.tif`
printf "$con_idiff \n"





