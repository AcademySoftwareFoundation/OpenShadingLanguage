#!/bin/csh

oslc test_cross_w16dvw16dvw16dv.osl  
oslc test_cross_w16dvw16vw16dv.osl
oslc test_cross_w16dvw16dvw16v.osl   
oslc test_cross_w16vw16vw16v.osl


echo "\n"
echo "*******************"
echo "osl_cross_w16dvw16dvw16v"
echo "*******************"

testshade --batched -g 200 200 test_cross_w16dvw16dvw16v -od uint8 -o dcross dcross_w16dvw16dvw16v_out.tif\
                                            -o dxcross dxcross_w16dvw16dvw16v_out.tif -o dycross dycross_w16dvw16dvw16v_out.tif \
                                            -o mdcross mdcross_w16dvw16dvw16v_out.tif\
                                    -o mdxcross mdxcross_w16dvw16dvw16v_out.tif -o mdycross mdycross_w16dvw16dvw16v_out.tif 

testshade -g 200 200 test_cross_w16dvw16dvw16v -od uint8 -o dcross dcross_w16dvw16dvw16v_ref.tif \
-o dxcross dxcross_w16dvw16dvw16v_ref.tif -o dycross dycross_w16dvw16dvw16v_ref.tif \
-o mdcross mdcross_w16dvw16dvw16v_ref.tif \
-o mdxcross mdxcross_w16dvw16dvw16v_ref.tif -o mdycross mdycross_w16dvw16dvw16v_ref.tif 

idiff dcross_w16dvw16dvw16v_ref.tif dcross_w16dvw16dvw16v_out.tif 
idiff dxcross_w16dvw16dvw16v_ref.tif dxcross_w16dvw16dvw16v_out.tif
idiff dycross_w16dvw16dvw16v_ref.tif dycross_w16dvw16dvw16v_out.tif


idiff mdcross_w16dvw16dvw16v_ref.tif mdcross_w16dvw16dvw16v_out.tif 
idiff mdxcross_w16dvw16dvw16v_ref.tif mdxcross_w16dvw16dvw16v_out.tif
idiff mdycross_w16dvw16dvw16v_ref.tif mdycross_w16dvw16dvw16v_out.tif


#****************

echo "\n"
echo "*******************"
echo "osl_cross_w16dvw16vw16dv"
echo "*******************"


testshade --batched -g 200 200 test_cross_w16dvw16vw16dv -od uint8 -o dcross dcross_w16dvw16vw16dv_out.tif \
-o dxcross dxcross_w16dvw16vw16dv_out.tif -o dycross dycross_w16dvw16vw16dv_out.tif \
-o mdcross mdcross_w16dvw16vw16dv_out.tif \
-o mdxcross mdxcross_w16dvw16vw16dv_out.tif -o mdycross mdycross_w16dvw16vw16dv_out.tif

testshade -g 200 200 test_cross_w16dvw16vw16dv -od uint8 -o dcross dcross_w16dvw16vw16dv_ref.tif \
-o dxcross dxcross_w16dvw16vw16dv_ref.tif -o dycross dycross_w16dvw16vw16dv_ref.tif \
-o mdcross mdcross_w16dvw16vw16dv_ref.tif \
-o mdxcross mdxcross_w16dvw16vw16dv_ref.tif -o mdycross mdycross_w16dvw16vw16dv_ref.tif \

idiff dcross_w16dvw16vw16dv_ref.tif dcross_w16dvw16vw16dv_out.tif
idiff dxcross_w16dvw16vw16dv_ref.tif dxcross_w16dvw16vw16dv_out.tif
idiff dycross_w16dvw16vw16dv_ref.tif dycross_w16dvw16vw16dv_out.tif

idiff mdcross_w16dvw16vw16dv_ref.tif mdcross_w16dvw16vw16dv_out.tif
idiff mdxcross_w16dvw16vw16dv_ref.tif mdxcross_w16dvw16vw16dv_out.tif
idiff mdycross_w16dvw16vw16dv_ref.tif mdycross_w16dvw16vw16dv_out.tif


#****************


echo "\n"
echo "*******************"
echo "osl_cross_w16dvw16dvw16dv"
echo "*******************"

testshade --batched -g 200 200 test_cross_w16dvw16dvw16dv -od uint8 -o dcross dcross_w16dvw16dvw16dv_out.tif \
-o dxcross dxcross_w16dvw16dvw16dv_out.tif -o dycross dycross_w16dvw16dvw16dv_out.tif \
-o mdcross mdcross_w16dvw16dvw16dv_out.tif \
-o mdxcross mdxcross_w16dvw16dvw16dv_out.tif -o mdycross mdycross_w16dvw16dvw16dv_out.tif

testshade -g 200 200 test_cross_w16dvw16dvw16dv -od uint8 -o dcross dcross_w16dvw16dvw16dv_ref.tif \
-o dxcross dxcross_w16dvw16dvw16dv_ref.tif -o dycross dycross_w16dvw16dvw16dv_ref.tif \
-o mdcross mdcross_w16dvw16dvw16dv_ref.tif \
-o mdxcross mdxcross_w16dvw16dvw16dv_ref.tif -o mdycross mdycross_w16dvw16dvw16dv_ref.tif

idiff -fail 0.004 dcross_w16dvw16dvw16dv_ref.tif dcross_w16dvw16dvw16dv_out.tif
idiff dxcross_w16dvw16dvw16dv_ref.tif dxcross_w16dvw16dvw16dv_out.tif
idiff dycross_w16dvw16dvw16dv_ref.tif dycross_w16dvw16dvw16dv_out.tif

idiff -fail 0.004 mdcross_w16dvw16dvw16dv_ref.tif mdcross_w16dvw16dvw16dv_out.tif
idiff mdxcross_w16dvw16dvw16dv_ref.tif mdxcross_w16dvw16dvw16dv_out.tif
idiff mdycross_w16dvw16dvw16dv_ref.tif mdycross_w16dvw16dvw16dv_out.tif


echo "\n"
echo "*******************"
echo "osl_cross_w16vw16vw16v"
echo "*******************"

testshade --batched -g 200 200 test_cross_w16vw16vw16v -o dcross dcross_w16vw16vw16v_out.tif -o mdcross mdcross_w16vw16vw16v_out.tif

testshade -g 200 200 test_cross_w16vw16vw16v  -o dcross dcross_w16vw16vw16v_ref.tif -o mdcross mdcross_w16vw16vw16v_ref.tif

idiff dcross_w16vw16vw16v_ref.tif dcross_w16vw16vw16v_out.tif
idiff mdcross_w16vw16vw16v_ref.tif mdcross_w16vw16vw16v_out.tif
