#!/bin/csh


oslc test_sincos_w16v_w16v_w16v.osl
oslc test_sincos_w16f_w16f_w16f.osl
oslc test_sincos_w16dv_w16v_w16dv.osl
oslc test_sincos_w16dv_w16dv_w16v.osl
oslc test_sincos_w16dv_w16dv_w16dv.osl
oslc test_sincos_w16df_w16f_w16df.osl
oslc test_sincos_w16df_w16df_w16f.osl
oslc test_sincos_w16df_w16df_w16df.osl
oslc test_sincos_w16f_f_f.osl


echo "\n"
echo "*******************"
echo "osl_sincos_w16f_f_f"
echo "*******************"

//
//testshade --batched -g 400 400 test_sincos_w16f_f_f -od uint8 -o sinop sinop_wfff_out.tif -o cosop cosop_wfff_out.tif \
//-o msinop msinop_wfff_out.tif -o mcosop mcosop_wfff_out.tif
//
//testshade --batched -g 400 400 test_sincos_w16f_f_f -od uint8 -o sinop sinop_wfff_ref.tif -o cosop cosop_wfff_ref.tif \
//-o msinop msinop_wfff_ref.tif -o mcosop mcosop_wfff_ref.tif
//
//idiff sinop_wfff_ref.tif sinop_wfff_out.tif
//idiff cosop_wfff_ref.tif cosop_wfff_out.tif
//idiff msinop_wfff_ref.tif msinop_wfff_out.tif
//idiff mcosop_wfff_ref.tif mcosop_wfff_out.tif

echo "\n"
echo "*******************"
echo "osl_sincos_w16f_w16f_w16f"
echo "*******************"


testshade --batched -g 400 400 test_sincos_w16f_w16f_w16f -od uint8 -o sinop sinop_wfwfwf_out.tif -o cosop cosop_wfwfwf_out.tif \
-o msinop msinop_wfwfwf_out.tif -o mcosop mcosop_wfwfwf_out.tif

testshade --batched -g 400 400 test_sincos_w16f_w16f_w16f -od uint8 -o sinop sinop_wfwfwf_ref.tif -o cosop cosop_wfwfwf_ref.tif \
-o msinop msinop_wfwfwf_ref.tif -o mcosop mcosop_wfwfwf_ref.tif

idiff sinop_wfwfwf_ref.tif sinop_wfwfwf_out.tif
idiff cosop_wfwfwf_ref.tif cosop_wfwfwf_out.tif
idiff msinop_wfwfwf_ref.tif msinop_wfwfwf_out.tif
idiff mcosop_wfwfwf_ref.tif mcosop_wfwfwf_out.tif





echo "\n"
echo "*******************"
echo "osl_sincos_w16df_w16df_w16df"
echo "*******************"



testshade --batched -g 400 400 test_sincos_w16df_w16df_w16df -od uint8 -o sinop sinop_wdfwdfwdf_out.tif \
                -o cosop cosop_wdfwdfwdf_out.tif -o dxsin dxsin_wdfwdfwdf_out.tif -o dxcos dxcos_wdfwdfwdf_out.tif -o dxa dxa_wdfwdfwdf_out.tif \
                -o msinop msinop_wdfwdfwdf_out.tif -o mcosop mcosop_wdfwdfwdf_out.tif -o mdxsin mdxsin_wdfwdfwdf_out.tif \
                -o mdxcos mdxcos_wdfwdfwdf_out.tif -o mdxa mdxa_wdfwdfwdf_out.tif

                testshade -g 400 400 test_sincos_w16df_w16df_w16df -od uint8 -o sinop sinop_wdfwdfwdf_ref.tif \
                                -o cosop cosop_wdfwdfwdf_ref.tif -o dxsin dxsin_wdfwdfwdf_ref.tif -o dxcos dxcos_wdfwdfwdf_ref.tif -o dxa dxa_wdfwdfwdf_ref.tif \
                                -o msinop msinop_wdfwdfwdf_ref.tif -o mcosop mcosop_wdfwdfwdf_ref.tif -o mdxsin mdxsin_wdfwdfwdf_ref.tif \
                                -o mdxcos mdxcos_wdfwdfwdf_ref.tif -o mdxa mdxa_wdfwdfwdf_ref.tif  
                                

idiff sinop_wdfwdfwdf_ref.tif sinop_wdfwdfwdf_out.tif
idiff cosop_wdfwdfwdf_ref.tif cosop_wdfwdfwdf_out.tif
idiff dxsin_wdfwdfwdf_ref.tif dxsin_wdfwdfwdf_out.tif
idiff dxcos_wdfwdfwdf_ref.tif dxcos_wdfwdfwdf_out.tif
idiff dxa_wdfwdfwdf_ref.tif dxa_wdfwdfwdf_out.tif



#Masked tests

idiff msinop_wdfwdfwdf_ref.tif msinop_wdfwdfwdf_out.tif
idiff mcosop_wdfwdfwdf_ref.tif mcosop_wdfwdfwdf_out.tif
idiff mdxsin_wdfwdfwdf_ref.tif mdxsin_wdfwdfwdf_out.tif
idiff mdxcos_wdfwdfwdf_ref.tif mdxcos_wdfwdfwdf_out.tif
idiff mdxa_wdfwdfwdf_ref.tif mdxa_wdfwdfwdf_out.tif




echo "\n"
echo "*******************"
echo "osl_sincos_w16df_w16df_w16f"
echo "*******************"



testshade --batched -g 400 400 test_sincos_w16df_w16df_w16f -od uint8 -o sinop sinop_wdfwdfwf_out.tif \
                -o cosop cosop_wdfwdfwf_out.tif -o dxsin dxsin_wdfwdfwf_out.tif  \
                -o msinop msinop_wdfwdfwf_out.tif -o mcosop mcosop_wdfwdfwf_out.tif -o mdxsin mdxsin_wdfwdfwf_out.tif
                

testshade -g 400 400 test_sincos_w16df_w16df_w16f -od uint8 -o sinop sinop_wdfwdfwf_ref.tif \
                                -o cosop cosop_wdfwdfwf_ref.tif -o dxsin dxsin_wdfwdfwf_ref.tif  \
                                -o msinop msinop_wdfwdfwf_ref.tif -o mcosop mcosop_wdfwdfwf_ref.tif -o mdxsin mdxsin_wdfwdfwf_ref.tif
                                
                                

idiff sinop_wdfwdfwf_ref.tif sinop_wdfwdfwf_out.tif
idiff cosop_wdfwdfwf_ref.tif cosop_wdfwdfwf_out.tif
idiff dxsin_wdfwdfwf_ref.tif dxsin_wdfwdfwf_out.tif



#Masked tests

idiff msinop_wdfwdfwf_ref.tif msinop_wdfwdfwf_out.tif
idiff mcosop_wdfwdfwf_ref.tif mcosop_wdfwdfwf_out.tif
idiff mdxsin_wdfwdfwf_ref.tif mdxsin_wdfwdfwf_out.tif



echo "\n"
echo "*******************"
echo "osl_sincos_w16df_w16f_w16df"
echo "*******************"

testshade --batched -g 400 400 test_sincos_w16df_w16f_w16df -od uint8 -o sinop sinop_wdfwfwdf_out.tif \
                -o cosop cosop_wdfwfwdf_out.tif -o dxcos dxcos_wdfwfwdf_out.tif  \
                -o msinop msinop_wdfwfwdf_out.tif -o mcosop mcosop_wdfwfwdf_out.tif -o mdxcos mdxcos_wdfwfwdf_out.tif
                

testshade -g 400 400 test_sincos_w16df_w16f_w16df -od uint8 -o sinop sinop_wdfwfwdf_ref.tif \
                                -o cosop cosop_wdfwfwdf_ref.tif -o dxcos dxcos_wdfwfwdf_ref.tif  \
                                -o msinop msinop_wdfwfwdf_ref.tif -o mcosop mcosop_wdfwfwdf_ref.tif -o mdxcos mdxcos_wdfwfwdf_ref.tif
                                
                                

idiff sinop_wdfwfwdf_ref.tif sinop_wdfwfwdf_out.tif
idiff cosop_wdfwfwdf_ref.tif cosop_wdfwfwdf_out.tif
idiff dxcos_wdfwfwdf_ref.tif dxcos_wdfwfwdf_out.tif



#Masked tests

idiff msinop_wdfwfwdf_ref.tif msinop_wdfwfwdf_out.tif
idiff mcosop_wdfwfwdf_ref.tif mcosop_wdfwfwdf_out.tif
idiff mdxcos_wdfwfwdf_ref.tif mdxcos_wdfwfwdf_out.tif


echo "\n"
echo "*******************"
echo "osl_sincos_w16v_w16v_w16v"
echo "*******************"


testshade --batched -g 400 400 test_sincos_w16v_w16v_w16v -od uint8 -o sinop sinop_wvwvwv_out.tif -o cosop cosop_wvwvwv_out.tif \
-o msinop msinop_wvwvwv_out.tif -o mcosop mcosop_wvwvwv_out.tif

testshade --batched -g 400 400 test_sincos_w16v_w16v_w16v -od uint8 -o sinop sinop_wvwvwv_ref.tif -o cosop cosop_wvwvwv_ref.tif \
-o msinop msinop_wvwvwv_ref.tif -o mcosop mcosop_wvwvwv_ref.tif

idiff sinop_wvwvwv_ref.tif sinop_wvwvwv_out.tif
idiff cosop_wvwvwv_ref.tif cosop_wvwvwv_out.tif
idiff msinop_wvwvwv_ref.tif msinop_wvwvwv_out.tif
idiff mcosop_wvwvwv_ref.tif mcosop_wvwvwv_out.tif




echo "\n"
echo "*******************"
echo "osl_sincos_w16dv_w16dv_w16dv"
echo "*******************"


testshade --batched -g 400 400 test_sincos_w16dv_w16dv_w16dv -od uint8 -o sinop sinop_wdvwdvwdv_out.tif \
                -o cosop cosop_wdvwdvwdv_out.tif -o dxsin dxsin_wdvwdvwdv_out.tif -o dxcos dxcos_wdvwdvwdv_out.tif -o dxa dxa_wdvwdvwdv_out.tif \
                -o msinop msinop_wdvwdvwdv_out.tif -o mcosop mcosop_wdvwdvwdv_out.tif  \
                -o mdxsin mdxsin_wdvwdvwdv_out.tif -o mdxcos mdxcos_wdvwdvwdv_out.tif -o mdxa mdxa_wdvwdvwdv_out.tif

                testshade -g 400 400 test_sincos_w16dv_w16dv_w16dv -od uint8 -o sinop sinop_wdvwdvwdv_ref.tif \
                                -o cosop cosop_wdvwdvwdv_ref.tif -o dxsin dxsin_wdvwdvwdv_ref.tif -o dxcos dxcos_wdvwdvwdv_ref.tif -o dxa dxa_wdvwdvwdv_ref.tif \
                                -o msinop msinop_wdvwdvwdv_ref.tif -o mcosop mcosop_wdvwdvwdv_ref.tif -o mdxsin mdxsin_wdvwdvwdv_ref.tif \
                                -o mdxcos mdxcos_wdvwdvwdv_ref.tif -o mdxa mdxa_wdvwdvwdv_ref.tif 
                                

idiff sinop_wdvwdvwdv_ref.tif sinop_wdvwdvwdv_out.tif
idiff cosop_wdvwdvwdv_ref.tif cosop_wdvwdvwdv_out.tif
idiff dxsin_wdvwdvwdv_ref.tif dxsin_wdvwdvwdv_out.tif
idiff dxcos_wdvwdvwdv_ref.tif dxcos_wdvwdvwdv_out.tif
idiff dxa_wdvwdvwdv_ref.tif dxa_wdvwdvwdv_out.tif


#Masked tests

idiff msinop_wdvwdvwdv_ref.tif msinop_wdvwdvwdv_out.tif
idiff mcosop_wdvwdvwdv_ref.tif mcosop_wdvwdvwdv_out.tif
idiff mdxsin_wdvwdvwdv_ref.tif mdxsin_wdvwdvwdv_out.tif
idiff mdxcos_wdvwdvwdv_ref.tif mdxcos_wdvwdvwdv_out.tif
idiff mdxa_wdvwdvwdv_ref.tif mdxa_wdvwdvwdv_out.tif



echo "\n"
echo "*******************"
echo "osl_sincos_w16dv_w16dv_w16v"
echo "*******************"


testshade --batched -g 400 400 test_sincos_w16dv_w16dv_w16v -od uint8 -o sinop sinop_wdvwdvwv_out.tif \
                -o cosop cosop_wdvwdvwv_out.tif -o dxsin dxsin_wdvwdvwv_out.tif  -o dxa dxa_wdvwdvwv_out.tif\
                -o msinop msinop_wdvwdvwv_out.tif -o mcosop mcosop_wdvwdvwv_out.tif -o mdxsin mdxsin_wdvwdvwv_out.tif -o mdxa mdxa_wdvwdvwv_out.tif
                

testshade -g 400 400 test_sincos_w16dv_w16dv_w16v -od uint8 -o sinop sinop_wdvwdvwv_ref.tif \
                                -o cosop cosop_wdvwdvwv_ref.tif -o dxsin dxsin_wdvwdvwv_ref.tif -o dxa dxa_wdvwdvwv_ref.tif \
                                -o msinop msinop_wdvwdvwv_ref.tif -o mcosop mcosop_wdvwdvwv_ref.tif -o mdxsin mdxsin_wdvwdvwv_ref.tif -o mdxa mdxa_wdvwdvwv_ref.tif 
                                
                                

idiff sinop_wdvwdvwv_ref.tif sinop_wdvwdvwv_out.tif
idiff cosop_wdvwdvwv_ref.tif cosop_wdvwdvwv_out.tif
idiff dxsin_wdvwdvwv_ref.tif dxsin_wdvwdvwv_out.tif
idiff dxa_wdvwdvwv_ref.tif dxa_wdvwdvwv_out.tif



#Masked tests

idiff msinop_wdvwdvwv_ref.tif msinop_wdvwdvwv_out.tif
idiff mcosop_wdvwdvwv_ref.tif mcosop_wdvwdvwv_out.tif
idiff mdxsin_wdvwdvwv_ref.tif mdxsin_wdvwdvwv_out.tif
idiff mdxa_wdvwdvwv_ref.tif mdxa_wdvwdvwv_out.tif




echo "\n"
echo "*******************"
echo "osl_sincos_w16dv_w16v_w16dv"
echo "*******************"



testshade --batched -g 400 400 test_sincos_w16dv_w16v_w16dv -od uint8 -o sinop sinop_wdvwvwdv_out.tif \
                -o cosop cosop_wdvwvwdv_out.tif  -o dxcos dxcos_wdvwvwdv_out.tif -o dxa dxa_wdvwvwdv_out.tif \
                -o msinop msinop_wdvwvwdv_out.tif -o mcosop mcosop_wdvwvwdv_out.tif \
                -o mdxcos mdxcos_wdvwvwdv_out.tif -o mdxa mdxa_wdvwvwdv_out.tif

testshade -g 400 400 test_sincos_w16dv_w16v_w16dv -od uint8 -o sinop sinop_wdvwvwdv_ref.tif \
                                -o cosop cosop_wdvwvwdv_ref.tif  -o dxcos dxcos_wdvwvwdv_ref.tif -o dxa dxa_wdvwvwdv_ref.tif \
                                -o msinop msinop_wdvwvwdv_ref.tif -o mcosop mcosop_wdvwvwdv_ref.tif  \
                                -o mdxcos mdxcos_wdvwvwdv_ref.tif -o mdxa mdxa_wdvwvwdv_ref.tif  
                                

idiff sinop_wdvwvwdv_ref.tif sinop_wdvwvwdv_out.tif
idiff cosop_wdvwvwdv_ref.tif cosop_wdvwvwdv_out.tif
idiff dxcos_wdvwvwdv_ref.tif dxcos_wdvwvwdv_out.tif
idiff dxa_wdvwvwdv_ref.tif dxa_wdvwvwdv_out.tif




idiff msinop_wdvwvwdv_ref.tif msinop_wdvwvwdv_out.tif
idiff mcosop_wdvwvwdv_ref.tif mcosop_wdvwvwdv_out.tif
idiff mdxcos_wdvwvwdv_ref.tif mdxcos_wdvwvwdv_out.tif
idiff mdxa_wdvwvwdv_ref.tif mdxa_wdvwvwdv_out.tif


