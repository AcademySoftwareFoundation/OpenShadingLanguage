rm *.tif *.oso

# color float includes masking
oslc test_color_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_color_u_float.tif test_color_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_color_u_float.tif test_color_u_float
idiff sout_color_u_float.tif bout_color_u_float.tif

oslc test_color_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_color_v_float.tif test_color_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_color_v_float.tif test_color_v_float
idiff sout_color_v_float.tif bout_color_v_float.tif


# Derivs includes masking
oslc test_color_v_dfloat.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_color_v_dfloat.tif test_color_v_dfloat
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_color_v_dfloat.tif test_color_v_dfloat
idiff sout_color_v_dfloat.tif bout_color_v_dfloat.tif





# color 3xfloat includes masking
oslc test_color_3xu_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_color_3xu_float.tif test_color_3xu_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_color_3xu_float.tif test_color_3xu_float
idiff sout_color_3xu_float.tif bout_color_3xu_float.tif

oslc test_color_3xv_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_color_3xv_float.tif test_color_3xv_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_color_3xv_float.tif test_color_3xv_float
idiff sout_color_3xv_float.tif bout_color_3xv_float.tif


# Derivs includes masking
oslc test_color_3xv_dfloat.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_color_3xv_dfloat.tif test_color_3xv_dfloat
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_color_3xv_dfloat.tif test_color_3xv_dfloat
idiff sout_color_3xv_dfloat.tif bout_color_3xv_dfloat.tif




# color colorspace 3x u float includes masking
oslc test_color_space_3xu_float.osl
testshade -t 1 -g 64 64 -param colorspace rgb -od uint8 -o Cout sout_rgb_3xu_float.tif test_color_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace rgb -od uint8 -o Cout bout_rgb_3xu_float.tif test_color_space_3xu_float
idiff sout_rgb_3xu_float.tif bout_rgb_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace RGB -od uint8 -o Cout sout_RGB_3xu_float.tif test_color_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace RGB -od uint8 -o Cout bout_RGB_3xu_float.tif test_color_space_3xu_float
idiff sout_RGB_3xu_float.tif bout_RGB_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace hsv -od uint8 -o Cout sout_hsv_3xu_float.tif test_color_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace hsv -od uint8 -o Cout bout_hsv_3xu_float.tif test_color_space_3xu_float
idiff sout_hsv_3xu_float.tif bout_hsv_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace hsl -od uint8 -o Cout sout_hsl_3xu_float.tif test_color_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace hsl -od uint8 -o Cout bout_hsl_3xu_float.tif test_color_space_3xu_float
idiff sout_hsl_3xu_float.tif bout_hsl_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace YIQ -od uint8 -o Cout sout_YIQ_3xu_float.tif test_color_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace YIQ -od uint8 -o Cout bout_YIQ_3xu_float.tif test_color_space_3xu_float
idiff sout_YIQ_3xu_float.tif bout_YIQ_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace XYZ -od uint8 -o Cout sout_XYZ_3xu_float.tif test_color_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace XYZ -od uint8 -o Cout bout_XYZ_3xu_float.tif test_color_space_3xu_float
idiff sout_XYZ_3xu_float.tif bout_XYZ_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace xyY -od uint8 -o Cout sout_xyY_3xu_float.tif test_color_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace xyY -od uint8 -o Cout bout_xyY_3xu_float.tif test_color_space_3xu_float
idiff sout_xyY_3xu_float.tif bout_xyY_3xu_float.tif


# color colorspace 3x v float includes masking
oslc test_color_space_3xv_float.osl
testshade -t 1 -g 64 64 -param colorspace rgb -od uint8 -o Cout sout_rgb_3xv_float.tif test_color_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace rgb -od uint8 -o Cout bout_rgb_3xv_float.tif test_color_space_3xv_float
idiff sout_rgb_3xv_float.tif bout_rgb_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace RGB -od uint8 -o Cout sout_RGB_3xv_float.tif test_color_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace RGB -od uint8 -o Cout bout_RGB_3xv_float.tif test_color_space_3xv_float
idiff sout_RGB_3xv_float.tif bout_RGB_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace hsv -od uint8 -o Cout sout_hsv_3xv_float.tif test_color_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace hsv -od uint8 -o Cout bout_hsv_3xv_float.tif test_color_space_3xv_float
idiff sout_hsv_3xv_float.tif bout_hsv_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace hsl -od uint8 -o Cout sout_hsl_3xv_float.tif test_color_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace hsl -od uint8 -o Cout bout_hsl_3xv_float.tif test_color_space_3xv_float
idiff sout_hsl_3xv_float.tif bout_hsl_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace YIQ -od uint8 -o Cout sout_YIQ_3xv_float.tif test_color_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace YIQ -od uint8 -o Cout bout_YIQ_3xv_float.tif test_color_space_3xv_float
idiff sout_YIQ_3xv_float.tif bout_YIQ_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace XYZ -od uint8 -o Cout sout_XYZ_3xv_float.tif test_color_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace XYZ -od uint8 -o Cout bout_XYZ_3xv_float.tif test_color_space_3xv_float
idiff sout_XYZ_3xv_float.tif bout_XYZ_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace xyY -od uint8 -o Cout sout_xyY_3xv_float.tif test_color_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace xyY -od uint8 -o Cout bout_xyY_3xv_float.tif test_color_space_3xv_float
idiff sout_xyY_3xv_float.tif bout_xyY_3xv_float.tif


# color colorspace 3x v dual float includes masking
# NOTE: current single point impl just 0's the derivs out, tests are to make sure we don't miss a fix for that
#       So expect all deriv based outputs to be black images
oslc test_color_space_3xv_dfloat.osl
testshade -t 1 -g 64 64 -param colorspace rgb -od uint8 -o Cout sout_rgb_3xv_dfloat.tif test_color_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace rgb -od uint8 -o Cout bout_rgb_3xv_dfloat.tif test_color_space_3xv_dfloat
idiff sout_rgb_3xv_dfloat.tif bout_rgb_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace RGB -od uint8 -o Cout sout_RGB_3xv_dfloat.tif test_color_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace RGB -od uint8 -o Cout bout_RGB_3xv_dfloat.tif test_color_space_3xv_dfloat
idiff sout_RGB_3xv_dfloat.tif bout_RGB_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace hsv -od uint8 -o Cout sout_hsv_3xv_dfloat.tif test_color_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace hsv -od uint8 -o Cout bout_hsv_3xv_dfloat.tif test_color_space_3xv_dfloat
idiff sout_hsv_3xv_dfloat.tif bout_hsv_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace hsl -od uint8 -o Cout sout_hsl_3xv_dfloat.tif test_color_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace hsl -od uint8 -o Cout bout_hsl_3xv_dfloat.tif test_color_space_3xv_dfloat
idiff sout_hsl_3xv_dfloat.tif bout_hsl_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace YIQ -od uint8 -o Cout sout_YIQ_3xv_dfloat.tif test_color_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace YIQ -od uint8 -o Cout bout_YIQ_3xv_dfloat.tif test_color_space_3xv_dfloat
idiff sout_YIQ_3xv_dfloat.tif bout_YIQ_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace XYZ -od uint8 -o Cout sout_XYZ_3xv_dfloat.tif test_color_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace XYZ -od uint8 -o Cout bout_XYZ_3xv_dfloat.tif test_color_space_3xv_dfloat
idiff sout_XYZ_3xv_dfloat.tif bout_XYZ_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace xyY -od uint8 -o Cout sout_xyY_3xv_dfloat.tif test_color_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace xyY -od uint8 -o Cout bout_xyY_3xv_dfloat.tif test_color_space_3xv_dfloat
idiff sout_xyY_3xv_dfloat.tif bout_xyY_3xv_dfloat.tif


# color varying colorspace 3x u float includes masking
oslc test_color_v_space_3xu_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_space_3xu_float.tif test_color_v_space_3xu_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_space_3xu_float.tif test_color_v_space_3xu_float
idiff sout_v_space_3xu_float.tif bout_v_space_3xu_float.tif


# color varying colorspace 3x v float includes masking
oslc test_color_v_space_3xv_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_space_3xv_float.tif test_color_v_space_3xv_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_space_3xv_float.tif test_color_v_space_3xv_float
idiff sout_v_space_3xv_float.tif bout_v_space_3xv_float.tif

# color varying colorspace 3x v dual float includes masking
# NOTE: current single point impl just 0's the derivs out, tests are to make sure we don't miss a fix for that
#       So expect all deriv based outputs to be black images
oslc test_color_v_space_3xv_dfloat.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_space_3xv_dfloat.tif test_color_v_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_space_3xv_dfloat.tif test_color_v_space_3xv_dfloat
idiff sout_v_space_3xv_dfloat.tif bout_v_space_3xv_dfloat.tif



