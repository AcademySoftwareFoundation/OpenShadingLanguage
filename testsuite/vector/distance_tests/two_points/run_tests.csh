#!/bin/csh

oslc distance_u_vector_u_vector.osl  
oslc distance_v_vector_u_vector.osl
oslc distance_u_vector_v_vector.osl  
oslc distance_v_vector_v_vector.osl


echo "\n"
echo "******************************"
echo "Varying vector; Uniform vector"
echo "******************************"

testshade --batched -g 200 200 distance_v_vector_u_vector -od uint8 -o ddistance ddistance_w16dfw16dvw16v_out.tif \
                                            -o dxdistance dxdistance_w16dfw16dvw16v_out.tif -o dydistance dydistance_w16dfw16dvw16v_out.tif \
                                            -o mddistance mddistance_w16dfw16dvw16v_out.tif \
                                            -o mdxdistance mdxdistance_w16dfw16dvw16v_out.tif -o mdydistance mdydistance_w16dfw16dvw16v_out.tif 
 

testshade  -g 200 200 distance_v_vector_u_vector -od uint8 -o ddistance ddistance_w16dfw16dvw16v_ref.tif \
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
echo "******************************"
echo "Uniform vector; Varying vector"
echo "******************************"

testshade --batched -g 200 200 distance_u_vector_v_vector -od uint8 -o ddistance ddistance_w16dfw16vw16dv_out.tif\
                        -o dxdistance dxdistance_w16dfw16vw16dv_out.tif -o dydistance dydistance_w16dfw16vw16dv_out.tif\
                        -o mddistance mddistance_w16dfw16vw16dv_out.tif\
                        -o mdxdistance mdxdistance_w16dfw16vw16dv_out.tif -o mdydistance mdydistance_w16dfw16vw16dv_out.tif
 

testshade  -g 200 200 distance_u_vector_v_vector -od uint8 -o ddistance ddistance_w16dfw16vw16dv_ref.tif\
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
echo "******************************"
echo "Varying vector; Varying vector"
echo "******************************"

testshade --batched -g 200 200 distance_v_vector_v_vector -od uint8 -o ddistance ddistance_w16dfw16dvw16dv_out.tif \
        -o dxdistance dxdistance_w16dfw16dvw16dv_out.tif -o dydistance dydistance_w16dfw16dvw16dv_out.tif \
        -o mddistance mddistance_w16dfw16dvw16dv_out.tif \
        -o mdxdistance mdxdistance_w16dfw16dvw16dv_out.tif -o mdydistance mdydistance_w16dfw16dvw16dv_out.tif
 
testshade  -g 200 200 distance_v_vector_v_vector -od uint8 -o ddistance ddistance_w16dfw16dvw16dv_ref.tif \
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
echo "******************************"
echo "Uniform vector; Uniform vector"
echo "*******************************"

testshade --batched -g 200 200 distance_u_vector_u_vector -od uint8 -o ddistance ddistance_w16fw16vw16v_out.tif -o mddistance mddistance_w16fw16vw16v_out.tif

testshade  -g 200 200 distance_u_vector_u_vector -od uint8 -o ddistance ddistance_w16fw16vw16v_ref.tif -o mddistance mddistance_w16fw16vw16v_ref.tif 

idiff ddistance_w16fw16vw16v_out.tif ddistance_w16fw16vw16v_ref.tif
idiff mddistance_w16fw16vw16v_out.tif mddistance_w16fw16vw16v_ref.tif



