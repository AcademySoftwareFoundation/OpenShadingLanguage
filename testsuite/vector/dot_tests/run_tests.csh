#!/bin/csh

oslc dot_u_vector_u_vector.osl  
oslc dot_u_vector_v_vector.osl  
oslc dot_v_vector_u_vector.osl  
oslc dot_v_vector_v_vector.osl


echo "\n"
echo "******************************"
echo "Varying vector; Uniform vector"
echo "******************************"

testshade --batched -g 200 200 dot_v_vector_u_vector -od uint8 -o ddot ddot_w16dfw16dvw16v_out.tif \
                                            -o dxdot dxdot_w16dfw16dvw16v_out.tif -o dydot dydot_w16dfw16dvw16v_out.tif \
                                            -o mddot mddot_w16dfw16dvw16v_out.tif \
                                            -o mdxdot mdxdot_w16dfw16dvw16v_out.tif -o mdydot mdydot_w16dfw16dvw16v_out.tif 
 

testshade  -g 200 200 dot_v_vector_u_vector -od uint8 -o ddot ddot_w16dfw16dvw16v_ref.tif \
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
echo "Uniform vector; Varying vector"
echo "*******************"

testshade --batched -g 200 200 dot_u_vector_v_vector -od uint8 -o ddot ddot_w16dfw16vw16dv_out.tif\
                        -o dxdot dxdot_w16dfw16vw16dv_out.tif -o dydot dydot_w16dfw16vw16dv_out.tif\
                        -o mddot mddot_w16dfw16vw16dv_out.tif\
                        -o mdxdot mdxdot_w16dfw16vw16dv_out.tif -o mdydot mdydot_w16dfw16vw16dv_out.tif
 

testshade  -g 200 200 dot_u_vector_v_vector -od uint8 -o ddot ddot_w16dfw16vw16dv_ref.tif\
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
echo "Varying vector; Varying vector"
echo "*******************"

testshade --batched -g 200 200 dot_v_vector_v_vector -od uint8 -o ddot ddot_w16dfw16dvw16dv_out.tif \
        -o dxdot dxdot_w16dfw16dvw16dv_out.tif -o dydot dydot_w16dfw16dvw16dv_out.tif \
        -o mddot mddot_w16dfw16dvw16dv_out.tif \
        -o mdxdot mdxdot_w16dfw16dvw16dv_out.tif -o mdydot mdydot_w16dfw16dvw16dv_out.tif
 
testshade  -g 200 200 dot_v_vector_v_vector -od uint8 -o ddot ddot_w16dfw16dvw16dv_ref.tif \
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
echo "Uniform vector; Uniform vector"
echo "*******************"

testshade --batched -g 200 200 dot_u_vector_u_vector -od uint8 -o ddot ddot_w16fw16vw16v_out.tif -o mddot mddot_w16fw16vw16v_out.tif

testshade  -g 200 200 dot_u_vector_u_vector -od uint8 -o ddot ddot_w16fw16vw16v_ref.tif -o mddot mddot_w16fw16vw16v_ref.tif 

idiff ddot_w16fw16vw16v_out.tif ddot_w16fw16vw16v_ref.tif
idiff mddot_w16fw16vw16v_out.tif mddot_w16fw16vw16v_ref.tif



