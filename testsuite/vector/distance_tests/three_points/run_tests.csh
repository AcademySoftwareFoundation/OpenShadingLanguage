#!/bin/csh

oslc distance_v_vector_v_vector_v_vector.osl  
oslc distance_v_vector_v_vector_u_vector.osl  
oslc distance_v_vector_u_vector_v_vector.osl  

oslc distance_u_vector_u_vector_u_vector.osl
oslc distance_u_vector_v_vector_u_vector.osl
oslc distance_u_vector_u_vector_v_vector.osl



echo "\n"
echo "**********************************************"
echo "Varying vector; Varying vector; Varying vector"
echo "**********************************************"

testshade --batched -g 200 200 distance_v_vector_v_vector_v_vector -od uint8 -o ddistance ddistance_vvv_out.tif \
        -o dxdistance dxdistance_vvv_out.tif -o dydistance dydistance_vvv_out.tif \
        -o mddistance mddistance_vvv_out.tif \
        -o mdxdistance mdxdistance_vvv_out.tif -o mdydistance mdydistance_vvv_out.tif
 
testshade  -g 200 200 distance_v_vector_v_vector_v_vector -od uint8 -o ddistance ddistance_vvv_ref.tif \
            -o dxdistance dxdistance_vvv_ref.tif -o dydistance dydistance_vvv_ref.tif \
            -o mddistance mddistance_vvv_ref.tif \
            -o mdxdistance mdxdistance_vvv_ref.tif -o mdydistance mdydistance_vvv_ref.tif 


idiff ddistance_vvv_out.tif ddistance_vvv_ref.tif
idiff dxdistance_vvv_out.tif dxdistance_vvv_ref.tif
idiff dydistance_vvv_out.tif dydistance_vvv_ref.tif

idiff mddistance_vvv_out.tif mddistance_vvv_ref.tif
idiff mdxdistance_vvv_out.tif mdxdistance_vvv_ref.tif
idiff mdydistance_vvv_out.tif mdydistance_vvv_ref.tif

echo "\n"
echo "**********************************************"
echo "Varying vector; Varying vector; Uniform vector"
echo "**********************************************"

testshade --batched -g 200 200 distance_v_vector_v_vector_u_vector -od uint8 -o ddistance ddistance_vvu_out.tif \
        -o dxdistance dxdistance_vvu_out.tif -o dydistance dydistance_vvu_out.tif \
        -o mddistance mddistance_vvu_out.tif \
        -o mdxdistance mdxdistance_vvu_out.tif -o mdydistance mdydistance_vvu_out.tif
 
testshade  -g 200 200 distance_v_vector_v_vector_u_vector -od uint8 -o ddistance ddistance_vvu_ref.tif \
            -o dxdistance dxdistance_vvu_ref.tif -o dydistance dydistance_vvu_ref.tif \
            -o mddistance mddistance_vvu_ref.tif \
            -o mdxdistance mdxdistance_vvu_ref.tif -o mdydistance mdydistance_vvu_ref.tif 


idiff ddistance_vvu_out.tif ddistance_vvu_ref.tif
idiff dxdistance_vvu_out.tif dxdistance_vvu_ref.tif
idiff dydistance_vvu_out.tif dydistance_vvu_ref.tif

idiff mddistance_vvu_out.tif mddistance_vvu_ref.tif
idiff mdxdistance_vvu_out.tif mdxdistance_vvu_ref.tif
idiff mdydistance_vvu_out.tif mdydistance_vvu_ref.tif

echo "\n"
echo "**********************************************"
echo "Varying vector; Uniform vector; Varying vector"
echo "**********************************************"

testshade --batched -g 200 200 distance_v_vector_u_vector_v_vector -od uint8 -o ddistance ddistance_vuv_out.tif \
        -o dxdistance dxdistance_vuv_out.tif -o dydistance dydistance_vuv_out.tif \
        -o mddistance mddistance_vuv_out.tif \
        -o mdxdistance mdxdistance_vuv_out.tif -o mdydistance mdydistance_vuv_out.tif
 
testshade  -g 200 200 distance_v_vector_u_vector_v_vector -od uint8 -o ddistance ddistance_vuv_ref.tif \
            -o dxdistance dxdistance_vuv_ref.tif -o dydistance dydistance_vuv_ref.tif \
            -o mddistance mddistance_vuv_ref.tif \
            -o mdxdistance mdxdistance_vuv_ref.tif -o mdydistance mdydistance_vuv_ref.tif 


idiff ddistance_vuv_out.tif ddistance_vuv_ref.tif
idiff dxdistance_vuv_out.tif dxdistance_vuv_ref.tif
idiff dydistance_vuv_out.tif dydistance_vuv_ref.tif

idiff mddistance_vuv_out.tif mddistance_vuv_ref.tif
idiff mdxdistance_vuv_out.tif mdxdistance_vuv_ref.tif
idiff mdydistance_vuv_out.tif mdydistance_vuv_ref.tif


echo "\n"
echo "**********************************************"
echo "Uniform vector; Uniform vector; Uniform vector"
echo "**********************************************"

testshade --batched -g 200 200 distance_u_vector_u_vector_u_vector -od uint8 -o ddistance ddistance_uuu_out.tif \
        -o dxdistance dxdistance_uuu_out.tif -o dydistance dydistance_uuu_out.tif \
        -o mddistance mddistance_uuu_out.tif \
        -o mdxdistance mdxdistance_uuu_out.tif -o mdydistance mdydistance_uuu_out.tif
 
testshade  -g 200 200 distance_u_vector_u_vector_u_vector -od uint8 -o ddistance ddistance_uuu_ref.tif \
            -o dxdistance dxdistance_uuu_ref.tif -o dydistance dydistance_uuu_ref.tif \
            -o mddistance mddistance_uuu_ref.tif \
            -o mdxdistance mdxdistance_uuu_ref.tif -o mdydistance mdydistance_uuu_ref.tif 


idiff ddistance_uuu_out.tif ddistance_uuu_ref.tif
idiff dxdistance_uuu_out.tif dxdistance_uuu_ref.tif
idiff dydistance_uuu_out.tif dydistance_uuu_ref.tif

idiff mddistance_uuu_out.tif mddistance_uuu_ref.tif
idiff mdxdistance_uuu_out.tif mdxdistance_uuu_ref.tif
idiff mdydistance_uuu_out.tif mdydistance_uuu_ref.tif


echo "\n"
echo "**********************************************"
echo "Uniform vector; Varying vector; Uniform vector"
echo "**********************************************"

testshade --batched -g 200 200 distance_u_vector_v_vector_u_vector -od uint8 -o ddistance ddistance_uvu_out.tif \
        -o dxdistance dxdistance_uvu_out.tif -o dydistance dydistance_uvu_out.tif \
        -o mddistance mddistance_uvu_out.tif \
        -o mdxdistance mdxdistance_uvu_out.tif -o mdydistance mdydistance_uvu_out.tif
 
testshade  -g 200 200 distance_u_vector_v_vector_u_vector -od uint8 -o ddistance ddistance_uvu_ref.tif \
            -o dxdistance dxdistance_uvu_ref.tif -o dydistance dydistance_uvu_ref.tif \
            -o mddistance mddistance_uvu_ref.tif \
            -o mdxdistance mdxdistance_uvu_ref.tif -o mdydistance mdydistance_uvu_ref.tif 


idiff ddistance_uvu_out.tif ddistance_uvu_ref.tif
idiff dxdistance_uvu_out.tif dxdistance_uvu_ref.tif
idiff dydistance_uvu_out.tif dydistance_uvu_ref.tif

idiff mddistance_uvu_out.tif mddistance_uvu_ref.tif
idiff mdxdistance_uvu_out.tif mdxdistance_uvu_ref.tif
idiff mdydistance_uvu_out.tif mdydistance_uvu_ref.tif


echo "\n"
echo "**********************************************"
echo "Uniform vector; Uniform vector; Vector vector"
echo "**********************************************"

testshade --batched -g 200 200 distance_u_vector_u_vector_v_vector -od uint8 -o ddistance ddistance_uuv_out.tif \
        -o dxdistance dxdistance_uuv_out.tif -o dydistance dydistance_uuv_out.tif \
        -o mddistance mddistance_uuv_out.tif \
        -o mdxdistance mdxdistance_uuv_out.tif -o mdydistance mdydistance_uuv_out.tif
 
testshade  -g 200 200 distance_u_vector_u_vector_v_vector -od uint8 -o ddistance ddistance_uuv_ref.tif \
            -o dxdistance dxdistance_uuv_ref.tif -o dydistance dydistance_uuv_ref.tif \
            -o mddistance mddistance_uuv_ref.tif \
            -o mdxdistance mdxdistance_uuv_ref.tif -o mdydistance mdydistance_uuv_ref.tif 


idiff ddistance_uuv_out.tif ddistance_uuv_ref.tif
idiff dxdistance_uuv_out.tif dxdistance_uuv_ref.tif
idiff dydistance_uuv_out.tif dydistance_uuv_ref.tif

idiff mddistance_uuv_out.tif mddistance_uuv_ref.tif
idiff mdxdistance_uuv_out.tif mdxdistance_uuv_ref.tif
idiff mdydistance_uuv_out.tif mdydistance_uuv_ref.tif












