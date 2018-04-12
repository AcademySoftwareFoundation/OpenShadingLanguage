#!/bin/csh

oslc test_distance_w16dfw16dvw16dv.osl    
oslc test_distance_w16dfw16dvw16v.osl  
oslc test_distance_w16fw16vw16v.osl
oslc test_distance_w16dfw16vw16dv.osl


echo "\n"
echo "*******************"
echo "osl_distance_w16dfw16dvw16v"
echo "*******************"

testshade --batched -g 200 200 test_distance_w16dfw16dvw16v -od uint8 -o ddistance ddistance_w16dfw16dvw16v_out.tif \
                                            -o dxdistance dxdistance_w16dfw16dvw16v_out.tif -o dydistance dydistance_w16dfw16dvw16v_out.tif \
                                            -o mddistance mddistance_w16dfw16dvw16v_out.tif \
                                            -o mdxdistance mdxdistance_w16dfw16dvw16v_out.tif -o mdydistance mdydistance_w16dfw16dvw16v_out.tif 
 

testshade  -g 200 200 test_distance_w16dfw16dvw16v -od uint8 -o ddistance ddistance_w16dfw16dvw16v_ref.tif \
                                                -o dxdistance dxdistance_w16dfw16dvw16v_ref.tif -o dydistance dydistance_w16dfw16dvw16v_ref.tif \
                                                -o mddistance mddistance_w16dfw16dvw16v_ref.tif \
                                                -o mdxdistance mdxdistance_w16dfw16dvw16v_ref.tif -o mdydistance mdydistance_w16dfw16dvw16v_ref.tif

idiff ddistance_w16dfw16dvw16v_out.tif ddistance_w16dfw16dvw16v_ref.tif
idiff dxdistance_w16dfw16dvw16v_out.tif dxdistance_w16dfw16dvw16v_ref.tif
idiff dydistance_w16dfw16dvw16v_out.tif dydistance_w16dfw16dvw16v_ref.tif

idiff mddistance_w16dfw16dvw16v_out.tif mddistance_w16dfw16dvw16v_ref.tif
idiff mdxdistance_w16dfw16dvw16v_out.tif mdxdistance_w16dfw16dvw16v_ref.tif
idiff mdydistance_w16dfw16dvw16v_out.tif mdydistance_w16dfw16dvw16v_ref.tif




#****************

echo "\n"
echo "*******************"
echo "osl_distance_w16dfw16vw16dv"
echo "*******************"

testshade --batched -g 200 200 test_distance_w16dfw16vw16dv -od uint8 -o ddistance ddistance_w16dfw16vw16dv_out.tif\
                        -o dxdistance dxdistance_w16dfw16vw16dv_out.tif -o dydistance dydistance_w16dfw16vw16dv_out.tif\
                        -o mddistance mddistance_w16dfw16vw16dv_out.tif\
                        -o mdxdistance mdxdistance_w16dfw16vw16dv_out.tif -o mdydistance mdydistance_w16dfw16vw16dv_out.tif
 

testshade  -g 200 200 test_distance_w16dfw16vw16dv -od uint8 -o ddistance ddistance_w16dfw16vw16dv_ref.tif\
                    -o dxdistance dxdistance_w16dfw16vw16dv_ref.tif -o dydistance dydistance_w16dfw16vw16dv_ref.tif\
                    -o mddistance mddistance_w16dfw16vw16dv_ref.tif\
                    -o mdxdistance mdxdistance_w16dfw16vw16dv_ref.tif -o mdydistance mdydistance_w16dfw16vw16dv_ref.tif


idiff ddistance_w16dfw16vw16dv_out.tif ddistance_w16dfw16vw16dv_ref.tif
idiff dxdistance_w16dfw16vw16dv_out.tif dxdistance_w16dfw16vw16dv_ref.tif
idiff dydistance_w16dfw16vw16dv_out.tif dydistance_w16dfw16vw16dv_ref.tif

idiff mddistance_w16dfw16vw16dv_out.tif mddistance_w16dfw16vw16dv_ref.tif
idiff mdxdistance_w16dfw16vw16dv_out.tif mdxdistance_w16dfw16vw16dv_ref.tif
idiff mdydistance_w16dfw16vw16dv_out.tif mdydistance_w16dfw16vw16dv_ref.tif


#****************

echo "\n"
echo "*******************"
echo "osl_distance_w16dfw16dvw16dv"
echo "*******************"

testshade --batched -g 200 200 test_distance_w16dfw16dvw16dv -od uint8 -o ddistance ddistance_w16dfw16dvw16dv_out.tif \
        -o dxdistance dxdistance_w16dfw16dvw16dv_out.tif -o dydistance dydistance_w16dfw16dvw16dv_out.tif \
        -o mddistance mddistance_w16dfw16dvw16dv_out.tif \
        -o mdxdistance mdxdistance_w16dfw16dvw16dv_out.tif -o mdydistance mdydistance_w16dfw16dvw16dv_out.tif
 
testshade  -g 200 200 test_distance_w16dfw16dvw16dv -od uint8 -o ddistance ddistance_w16dfw16dvw16dv_ref.tif \
            -o dxdistance dxdistance_w16dfw16dvw16dv_ref.tif -o dydistance dydistance_w16dfw16dvw16dv_ref.tif \
            -o mddistance mddistance_w16dfw16dvw16dv_ref.tif \
            -o mdxdistance mdxdistance_w16dfw16dvw16dv_ref.tif -o mdydistance mdydistance_w16dfw16dvw16dv_ref.tif 


idiff ddistance_w16dfw16dvw16dv_out.tif ddistance_w16dfw16dvw16dv_ref.tif
idiff dxdistance_w16dfw16dvw16dv_out.tif dxdistance_w16dfw16dvw16dv_ref.tif
idiff dydistance_w16dfw16dvw16dv_out.tif dydistance_w16dfw16dvw16dv_ref.tif

idiff mddistance_w16dfw16dvw16dv_out.tif mddistance_w16dfw16dvw16dv_ref.tif
idiff mdxdistance_w16dfw16dvw16dv_out.tif mdxdistance_w16dfw16dvw16dv_ref.tif
idiff mdydistance_w16dfw16dvw16dv_out.tif mdydistance_w16dfw16dvw16dv_ref.tif


echo "\n"
echo "*******************"
echo "osl_distance_w16fw16vw16v"
echo "*******************"

testshade --batched -g 200 200 test_distance_w16fw16vw16v -od uint8 -o ddistance ddistance_w16fw16vw16v_out.tif -o mddistance mddistance_w16fw16vw16v_out.tif

testshade  -g 200 200 test_distance_w16fw16vw16v -od uint8 -o ddistance ddistance_w16fw16vw16v_ref.tif -o mddistance mddistance_w16fw16vw16v_ref.tif 

idiff ddistance_w16fw16vw16v_out.tif ddistance_w16fw16vw16v_ref.tif
idiff mddistance_w16fw16vw16v_out.tif mddistance_w16fw16vw16v_ref.tif



