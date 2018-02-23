#!/bin/csh

echo "\n"
echo "*******************"
echo "Catmull-Rom spline..."
echo "*******************"


set ITER_NUM=2

set cmr_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline catmullrom_vec3_out.tif  test_catmullrom_vec3 | grep "Run  :"`
printf "Spline (Vec3) Batched time: "
printf "$cmr_cmd \n"

set cmr_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline catmullrom_vec3_ref.tif  test_catmullrom_vec3 | grep "Run  :"`

printf "Non-batched time: "
printf "$cmr_cmd_ref\n"

set cmr_idiff=`idiff catmullrom_vec3_ref.tif catmullrom_vec3_out.tif`
printf "$cmr_idiff \n"



echo "\n"
echo "*******************"
echo "Bezier spline..."
echo "*******************"

set bez_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8  -o Fspline bezier_vec3_out.tif  test_bezier_vec3 | grep "Run  :"`
printf "Batched time: "
printf "$bez_cmd \n"

set bez_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline bezier_vec3_ref.tif  test_bezier_vec3 | grep "Run  :"`
printf "Non-batched time: "
printf "$bez_cmd_ref\n"

set bez_idiff=`idiff bezier_vec3_ref.tif bezier_vec3_out.tif`
printf "$bez_idiff \n"

echo "\n"
echo "*******************"
echo "Bspline spline..."
echo "*******************"

set bsp_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline bspline_vec3_out.tif  test_bspline_vec3 | grep "Run  :"`
printf "Batched time: "
printf "$bsp_cmd \n"

set bsp_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline bspline_vec3_ref.tif  test_bspline_vec3 | grep "Run  :"`
printf "Non-batched time: "
printf "$bsp_cmd_ref\n"

set bsp_idiff=`idiff bspline_vec3_ref.tif bspline_vec3_out.tif`
printf "$bsp_idiff \n"

echo "\n"
echo "*******************"
echo "Hermite spline..."
echo "*******************"

set her_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline hermite_vec3_out.tif  test_hermite_vec3 | grep "Run  :"`
printf "Batched time: "
printf "$her_cmd \n"

set her_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline hermite_vec3_ref.tif  test_hermite_vec3 | grep "Run  :"`

printf "Non-batched time: "
printf "$her_cmd_ref\n"

set her_idiff=`idiff hermite_vec3_ref.tif hermite_vec3_out.tif`
printf "$her_idiff \n"



echo "\n"
echo "*******************"
echo "Linear spline..."
echo "*******************"

set lin_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline linear_vec3_out.tif  test_linear_vec3 | grep "Run  :"`
printf "Batched time: "
printf "$lin_cmd \n"

set lin_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline linear_vec3_ref.tif  test_linear_vec3 | grep "Run  :"`

printf "Non-batched time: "
printf "$lin_cmd_ref\n"

set lin_idiff=`idiff linear_vec3_ref.tif linear_vec3_out.tif`
printf "$lin_idiff \n"


echo "\n"
echo "*******************"
echo "Constant spline..."
echo "*******************"

set con_cmd=`testshade --batched -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline constant_vec3_out.tif  test_constant_vec3 | grep "Run  :"`

printf "Batched time: "
printf "$con_cmd \n"

set con_cmd_ref=`testshade -g 1024 1024 -t 1 --iters $ITER_NUM --stats -od uint8 -o Fspline constant_vec3_ref.tif  test_constant_vec3 | grep "Run  :"`

printf "Non-batched time: "
printf "$con_cmd_ref\n"

set con_idiff=`idiff constant_vec3_ref.tif constant_vec3_out.tif`
printf "$con_idiff \n"


