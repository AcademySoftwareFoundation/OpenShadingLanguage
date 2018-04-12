rm *.tif *.oso

# luminance u color includes masking
oslc test_luminance_u_color.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_color_u_color.tif test_luminance_u_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_color_u_color.tif test_luminance_u_color
idiff sout_color_u_color.tif bout_color_u_color.tif


# luminance v color includes masking
oslc test_luminance_v_color.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_color_v_color.tif test_luminance_v_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_color_v_color.tif test_luminance_v_color
idiff sout_color_v_color.tif bout_color_v_color.tif


# luminance v derivatives of color includes masking
oslc test_luminance_v_dcolor.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_color_v_dcolor.tif test_luminance_v_dcolor
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_color_v_dcolor.tif test_luminance_v_dcolor
idiff sout_color_v_dcolor.tif bout_color_v_dcolor.tif
