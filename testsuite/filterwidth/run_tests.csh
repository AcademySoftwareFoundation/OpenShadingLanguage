#!/bin/csh

oslc test_filterwidth_w16f_w16df.osl
oslc test_filterwidth_w16v_w16dv.osl

echo "\n"
echo "*******************"
echo "osl_filterwidth_w16vw16dv"
echo "*******************"

testshade --batched -g 200 200 test_filterwidth_w16vw16dv -o op fw_w16vw16dv_out.tif -o dxop dxfw_w16vw16dv_out.tif \
                                                         -o mop mfw_w16vw16dv_out.tif -o mdxop mdxfw_w16vw16dv_out.tif 
                                                                         

testshade -g 200 200 test_filterwidth_w16vw16dv -o op fw_w16vw16dv_ref.tif -o dxop dxfw_w16vw16dv_ref.tif \
                                                          -o mop mfw_w16vw16dv_ref.tif -o mdxop mdxfw_w16vw16dv_ref.tif 

idiff fw_w16vw16dv_ref.tif fw_w16vw16dv_out.tif
idiff dxfw_w16vw16dv_ref.tif dxfw_w16vw16dv_out.tif


idiff mfw_w16vw16dv_ref.tif mfw_w16vw16dv_out.tif
idiff mdxfw_w16vw16dv_ref.tif mdxfw_w16vw16dv_out.tif


echo "\n"
echo "*******************"
echo "osl_filterwidth_w16fw16df"
echo "*******************"

testshade --batched -g 200 200 test_filterwidth_w16fw16df -o op fw_w16fw16df_out.tif  \
                                                         -o mop mfw_w16fw16df_out.tif 
                                                                         

testshade -g 200 200 test_filterwidth_w16fw16df -o op fw_w16fw16df_ref.tif \
                                                          -o mop mfw_w16fw16df_ref.tif 

idiff fw_w16fw16df_ref.tif fw_w16fw16df_out.tif
idiff mfw_w16fw16df_ref.tif mfw_w16fw16df_out.tif
