#!/bin/csh

oslc test_dot_w16dfw16dvw16dv.osl    
oslc test_dot_w16dfw16dvw16v.osl  
oslc test_dot_w16fw16vw16v.osl
oslc test_dot_w16dfw16vw16dv.osl


echo "\n"
echo "*******************"
echo "osl_dot_w16dfw16dvw16v"
echo "*******************"

testshade --batched -g 200 200 test_dot_w16dfw16dvw16v -od uint8 -o ddot ddot_w16dfw16dvw16v_out.tif \
                                            -o dxdot dxdot_w16dfw16dvw16v_out.tif -o dydot dydot_w16dfw16dvw16v_out.tif \
                                            -o mddot mddot_w16dfw16dvw16v_out.tif \
                                            -o mdxdot mdxdot_w16dfw16dvw16v_out.tif -o mdydot mdydot_w16dfw16dvw16v_out.tif 
 

testshade  -g 200 200 test_dot_w16dfw16dvw16v -od uint8 -o ddot ddot_w16dfw16dvw16v_ref.tif \
                                                -o dxdot dxdot_w16dfw16dvw16v_ref.tif -o dydot dydot_w16dfw16dvw16v_ref.tif \
                                                -o mddot mddot_w16dfw16dvw16v_ref.tif \
                                                -o mdxdot mdxdot_w16dfw16dvw16v_ref.tif -o mdydot mdydot_w16dfw16dvw16v_ref.tif

idiff ddot_w16dfw16dvw16v_out.tif ddot_w16dfw16dvw16v_ref.tif
idiff dxdot_w16dfw16dvw16v_out.tif dxdot_w16dfw16dvw16v_ref.tif
idiff dydot_w16dfw16dvw16v_out.tif dydot_w16dfw16dvw16v_ref.tif

idiff mddot_w16dfw16dvw16v_out.tif mddot_w16dfw16dvw16v_ref.tif
idiff mdxdot_w16dfw16dvw16v_out.tif mdxdot_w16dfw16dvw16v_ref.tif
idiff mdydot_w16dfw16dvw16v_out.tif mdydot_w16dfw16dvw16v_ref.tif




#****************

echo "\n"
echo "*******************"
echo "osl_dot_w16dfw16vw16dv"
echo "*******************"

testshade --batched -g 200 200 test_dot_w16dfw16vw16dv -od uint8 -o ddot ddot_w16dfw16vw16dv_out.tif\
                        -o dxdot dxdot_w16dfw16vw16dv_out.tif -o dydot dydot_w16dfw16vw16dv_out.tif\
                        -o mddot mddot_w16dfw16vw16dv_out.tif\
                        -o mdxdot mdxdot_w16dfw16vw16dv_out.tif -o mdydot mdydot_w16dfw16vw16dv_out.tif
 

testshade  -g 200 200 test_dot_w16dfw16vw16dv -od uint8 -o ddot ddot_w16dfw16vw16dv_ref.tif\
                    -o dxdot dxdot_w16dfw16vw16dv_ref.tif -o dydot dydot_w16dfw16vw16dv_ref.tif\
                    -o mddot mddot_w16dfw16vw16dv_ref.tif\
                    -o mdxdot mdxdot_w16dfw16vw16dv_ref.tif -o mdydot mdydot_w16dfw16vw16dv_ref.tif


idiff ddot_w16dfw16vw16dv_out.tif ddot_w16dfw16vw16dv_ref.tif
idiff dxdot_w16dfw16vw16dv_out.tif dxdot_w16dfw16vw16dv_ref.tif
idiff dydot_w16dfw16vw16dv_out.tif dydot_w16dfw16vw16dv_ref.tif

idiff mddot_w16dfw16vw16dv_out.tif mddot_w16dfw16vw16dv_ref.tif
idiff mdxdot_w16dfw16vw16dv_out.tif mdxdot_w16dfw16vw16dv_ref.tif
idiff mdydot_w16dfw16vw16dv_out.tif mdydot_w16dfw16vw16dv_ref.tif


#****************

echo "\n"
echo "*******************"
echo "osl_dot_w16dfw16dvw16dv"
echo "*******************"

testshade --batched -g 200 200 test_dot_w16dfw16dvw16dv -od uint8 -o ddot ddot_w16dfw16dvw16dv_out.tif \
        -o dxdot dxdot_w16dfw16dvw16dv_out.tif -o dydot dydot_w16dfw16dvw16dv_out.tif \
        -o mddot mddot_w16dfw16dvw16dv_out.tif \
        -o mdxdot mdxdot_w16dfw16dvw16dv_out.tif -o mdydot mdydot_w16dfw16dvw16dv_out.tif
 
testshade  -g 200 200 test_dot_w16dfw16dvw16dv -od uint8 -o ddot ddot_w16dfw16dvw16dv_ref.tif \
            -o dxdot dxdot_w16dfw16dvw16dv_ref.tif -o dydot dydot_w16dfw16dvw16dv_ref.tif \
            -o mddot mddot_w16dfw16dvw16dv_ref.tif \
            -o mdxdot mdxdot_w16dfw16dvw16dv_ref.tif -o mdydot mdydot_w16dfw16dvw16dv_ref.tif 


idiff ddot_w16dfw16dvw16dv_out.tif ddot_w16dfw16dvw16dv_ref.tif
idiff dxdot_w16dfw16dvw16dv_out.tif dxdot_w16dfw16dvw16dv_ref.tif
idiff dydot_w16dfw16dvw16dv_out.tif dydot_w16dfw16dvw16dv_ref.tif

idiff mddot_w16dfw16dvw16dv_out.tif mddot_w16dfw16dvw16dv_ref.tif
idiff mdxdot_w16dfw16dvw16dv_out.tif mdxdot_w16dfw16dvw16dv_ref.tif
idiff mdydot_w16dfw16dvw16dv_out.tif mdydot_w16dfw16dvw16dv_ref.tif


echo "\n"
echo "*******************"
echo "osl_dot_w16fw16vw16v"
echo "*******************"

testshade --batched -g 200 200 test_dot_w16fw16vw16v -od uint8 -o ddot ddot_w16fw16vw16v_out.tif -o mddot mddot_w16fw16vw16v_out.tif

testshade  -g 200 200 test_dot_w16fw16vw16v -od uint8 -o ddot ddot_w16fw16vw16v_ref.tif -o mddot mddot_w16fw16vw16v_ref.tif 

idiff ddot_w16fw16vw16v_out.tif ddot_w16fw16vw16v_ref.tif
idiff mddot_w16fw16vw16v_out.tif mddot_w16fw16vw16v_ref.tif



