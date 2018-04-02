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
oslc test_color_u_space_3xu_float.osl
testshade -t 1 -g 64 64 -param colorspace rgb -od uint8 -o Cout sout_rgb_3xu_float.tif test_color_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace rgb -od uint8 -o Cout bout_rgb_3xu_float.tif test_color_u_space_3xu_float
idiff sout_rgb_3xu_float.tif bout_rgb_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace RGB -od uint8 -o Cout sout_RGB_3xu_float.tif test_color_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace RGB -od uint8 -o Cout bout_RGB_3xu_float.tif test_color_u_space_3xu_float
idiff sout_RGB_3xu_float.tif bout_RGB_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace hsv -od uint8 -o Cout sout_hsv_3xu_float.tif test_color_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace hsv -od uint8 -o Cout bout_hsv_3xu_float.tif test_color_u_space_3xu_float
idiff sout_hsv_3xu_float.tif bout_hsv_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace hsl -od uint8 -o Cout sout_hsl_3xu_float.tif test_color_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace hsl -od uint8 -o Cout bout_hsl_3xu_float.tif test_color_u_space_3xu_float
idiff sout_hsl_3xu_float.tif bout_hsl_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace YIQ -od uint8 -o Cout sout_YIQ_3xu_float.tif test_color_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace YIQ -od uint8 -o Cout bout_YIQ_3xu_float.tif test_color_u_space_3xu_float
idiff sout_YIQ_3xu_float.tif bout_YIQ_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace XYZ -od uint8 -o Cout sout_XYZ_3xu_float.tif test_color_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace XYZ -od uint8 -o Cout bout_XYZ_3xu_float.tif test_color_u_space_3xu_float
idiff sout_XYZ_3xu_float.tif bout_XYZ_3xu_float.tif

testshade -t 1 -g 64 64 -param colorspace xyY -od uint8 -o Cout sout_xyY_3xu_float.tif test_color_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param colorspace xyY -od uint8 -o Cout bout_xyY_3xu_float.tif test_color_u_space_3xu_float
idiff sout_xyY_3xu_float.tif bout_xyY_3xu_float.tif


# color colorspace 3x v float includes masking
oslc test_color_u_space_3xv_float.osl
testshade -t 1 -g 64 64 -param colorspace rgb -od uint8 -o Cout sout_rgb_3xv_float.tif test_color_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace rgb -od uint8 -o Cout bout_rgb_3xv_float.tif test_color_u_space_3xv_float
idiff sout_rgb_3xv_float.tif bout_rgb_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace RGB -od uint8 -o Cout sout_RGB_3xv_float.tif test_color_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace RGB -od uint8 -o Cout bout_RGB_3xv_float.tif test_color_u_space_3xv_float
idiff sout_RGB_3xv_float.tif bout_RGB_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace hsv -od uint8 -o Cout sout_hsv_3xv_float.tif test_color_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace hsv -od uint8 -o Cout bout_hsv_3xv_float.tif test_color_u_space_3xv_float
idiff sout_hsv_3xv_float.tif bout_hsv_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace hsl -od uint8 -o Cout sout_hsl_3xv_float.tif test_color_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace hsl -od uint8 -o Cout bout_hsl_3xv_float.tif test_color_u_space_3xv_float
idiff sout_hsl_3xv_float.tif bout_hsl_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace YIQ -od uint8 -o Cout sout_YIQ_3xv_float.tif test_color_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace YIQ -od uint8 -o Cout bout_YIQ_3xv_float.tif test_color_u_space_3xv_float
idiff sout_YIQ_3xv_float.tif bout_YIQ_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace XYZ -od uint8 -o Cout sout_XYZ_3xv_float.tif test_color_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace XYZ -od uint8 -o Cout bout_XYZ_3xv_float.tif test_color_u_space_3xv_float
idiff sout_XYZ_3xv_float.tif bout_XYZ_3xv_float.tif

testshade -t 1 -g 64 64 -param colorspace xyY -od uint8 -o Cout sout_xyY_3xv_float.tif test_color_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param colorspace xyY -od uint8 -o Cout bout_xyY_3xv_float.tif test_color_u_space_3xv_float
idiff sout_xyY_3xv_float.tif bout_xyY_3xv_float.tif


# color colorspace 3x v dual float includes masking
# NOTE: current single point impl just 0's the derivs out, tests are to make sure we don't miss a fix for that
#       So expect all deriv based outputs to be black images
oslc test_color_u_space_3xv_dfloat.osl
testshade -t 1 -g 64 64 -param colorspace rgb -od uint8 -o Cout sout_rgb_3xv_dfloat.tif test_color_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace rgb -od uint8 -o Cout bout_rgb_3xv_dfloat.tif test_color_u_space_3xv_dfloat
idiff sout_rgb_3xv_dfloat.tif bout_rgb_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace RGB -od uint8 -o Cout sout_RGB_3xv_dfloat.tif test_color_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace RGB -od uint8 -o Cout bout_RGB_3xv_dfloat.tif test_color_u_space_3xv_dfloat
idiff sout_RGB_3xv_dfloat.tif bout_RGB_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace hsv -od uint8 -o Cout sout_hsv_3xv_dfloat.tif test_color_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace hsv -od uint8 -o Cout bout_hsv_3xv_dfloat.tif test_color_u_space_3xv_dfloat
idiff sout_hsv_3xv_dfloat.tif bout_hsv_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace hsl -od uint8 -o Cout sout_hsl_3xv_dfloat.tif test_color_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace hsl -od uint8 -o Cout bout_hsl_3xv_dfloat.tif test_color_u_space_3xv_dfloat
idiff sout_hsl_3xv_dfloat.tif bout_hsl_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace YIQ -od uint8 -o Cout sout_YIQ_3xv_dfloat.tif test_color_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace YIQ -od uint8 -o Cout bout_YIQ_3xv_dfloat.tif test_color_u_space_3xv_dfloat
idiff sout_YIQ_3xv_dfloat.tif bout_YIQ_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace XYZ -od uint8 -o Cout sout_XYZ_3xv_dfloat.tif test_color_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace XYZ -od uint8 -o Cout bout_XYZ_3xv_dfloat.tif test_color_u_space_3xv_dfloat
idiff sout_XYZ_3xv_dfloat.tif bout_XYZ_3xv_dfloat.tif

testshade -t 1 -g 64 64 -param colorspace xyY -od uint8 -o Cout sout_xyY_3xv_dfloat.tif test_color_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -param colorspace xyY -od uint8 -o Cout bout_xyY_3xv_dfloat.tif test_color_u_space_3xv_dfloat
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




# transformc u colorspace u color includes masking
oslc test_transformc_u_space_u_color.osl
testshade -t 1 -g 64 64 -param colorspace rgb -od uint8 -o Cout sout_transformc_rgb_u_color.tif test_transformc_u_space_u_color
testshade -t 1 --batched -g 64 64 -param colorspace rgb -od uint8 -o Cout bout_transformc_rgb_u_color.tif test_transformc_u_space_u_color
idiff sout_transformc_rgb_u_color.tif bout_transformc_rgb_u_color.tif

testshade -t 1 -g 64 64 -param colorspace RGB -od uint8 -o Cout sout_transformc_RGB_u_color.tif test_transformc_u_space_u_color
testshade -t 1 --batched -g 64 64 -param colorspace RGB -od uint8 -o Cout bout_transformc_RGB_u_color.tif test_transformc_u_space_u_color
idiff sout_transformc_RGB_u_color.tif bout_transformc_RGB_u_color.tif

testshade -t 1 -g 64 64 -param colorspace hsv -od uint8 -o Cout sout_transformc_hsv_u_color.tif test_transformc_u_space_u_color
testshade -t 1 --batched -g 64 64 -param colorspace hsv -od uint8 -o Cout bout_transformc_hsv_u_color.tif test_transformc_u_space_u_color
idiff sout_transformc_hsv_u_color.tif bout_transformc_hsv_u_color.tif

testshade -t 1 -g 64 64 -param colorspace hsl -od uint8 -o Cout sout_transformc_hsl_u_color.tif test_transformc_u_space_u_color
testshade -t 1 --batched -g 64 64 -param colorspace hsl -od uint8 -o Cout bout_transformc_hsl_u_color.tif test_transformc_u_space_u_color
idiff sout_transformc_hsl_u_color.tif bout_transformc_hsl_u_color.tif

testshade -t 1 -g 64 64 -param colorspace YIQ -od uint8 -o Cout sout_transformc_YIQ_u_color.tif test_transformc_u_space_u_color
testshade -t 1 --batched -g 64 64 -param colorspace YIQ -od uint8 -o Cout bout_transformc_YIQ_u_color.tif test_transformc_u_space_u_color
idiff sout_transformc_YIQ_u_color.tif bout_transformc_YIQ_u_color.tif

testshade -t 1 -g 64 64 -param colorspace XYZ -od uint8 -o Cout sout_transformc_XYZ_u_color.tif test_transformc_u_space_u_color
testshade -t 1 --batched -g 64 64 -param colorspace XYZ -od uint8 -o Cout bout_transformc_XYZ_u_color.tif test_transformc_u_space_u_color
idiff sout_transformc_XYZ_u_color.tif bout_transformc_XYZ_u_color.tif

# TODO: stdosl.h implementation doesn't support colorspace of xyY, an oversight/bug?
#testshade -t 1 -g 64 64 -param colorspace xyY -od uint8 -o Cout sout_transformc_xyY_u_color.tif test_transformc_u_space_u_color
#testshade -t 1 --batched -g 64 64 -param colorspace xyY -od uint8 -o Cout bout_transformc_xyY_u_color.tif test_transformc_u_space_u_color
#idiff sout_transformc_xyY_u_color.tif bout_transformc_xyY_u_color.tif


# transformc u colorspace v color includes masking
oslc test_transformc_u_space_v_color.osl
testshade -t 1 -g 64 64 -param colorspace rgb -od uint8 -o Cout sout_transformc_rgb_v_color.tif test_transformc_u_space_v_color
testshade -t 1 --batched -g 64 64 -param colorspace rgb -od uint8 -o Cout bout_transformc_rgb_v_color.tif test_transformc_u_space_v_color
idiff sout_transformc_rgb_v_color.tif bout_transformc_rgb_v_color.tif

testshade -t 1 -g 64 64 -param colorspace RGB -od uint8 -o Cout sout_transformc_RGB_v_color.tif test_transformc_u_space_v_color
testshade -t 1 --batched -g 64 64 -param colorspace RGB -od uint8 -o Cout bout_transformc_RGB_v_color.tif test_transformc_u_space_v_color
idiff sout_transformc_RGB_v_color.tif bout_transformc_RGB_v_color.tif

testshade -t 1 -g 64 64 -param colorspace hsv -od uint8 -o Cout sout_transformc_hsv_v_color.tif test_transformc_u_space_v_color
testshade -t 1 --batched -g 64 64 -param colorspace hsv -od uint8 -o Cout bout_transformc_hsv_v_color.tif test_transformc_u_space_v_color
idiff sout_transformc_hsv_v_color.tif bout_transformc_hsv_v_color.tif

testshade -t 1 -g 64 64 -param colorspace hsl -od uint8 -o Cout sout_transformc_hsl_v_color.tif test_transformc_u_space_v_color
testshade -t 1 --batched -g 64 64 -param colorspace hsl -od uint8 -o Cout bout_transformc_hsl_v_color.tif test_transformc_u_space_v_color
idiff -fail 0.004 sout_transformc_hsl_v_color.tif bout_transformc_hsl_v_color.tif

testshade -t 1 -g 64 64 -param colorspace YIQ -od uint8 -o Cout sout_transformc_YIQ_v_color.tif test_transformc_u_space_v_color
testshade -t 1 --batched -g 64 64 -param colorspace YIQ -od uint8 -o Cout bout_transformc_YIQ_v_color.tif test_transformc_u_space_v_color
idiff sout_transformc_YIQ_v_color.tif bout_transformc_YIQ_v_color.tif

testshade -t 1 -g 64 64 -param colorspace XYZ -od uint8 -o Cout sout_transformc_XYZ_v_color.tif test_transformc_u_space_v_color
testshade -t 1 --batched -g 64 64 -param colorspace XYZ -od uint8 -o Cout bout_transformc_XYZ_v_color.tif test_transformc_u_space_v_color
idiff sout_transformc_XYZ_v_color.tif bout_transformc_XYZ_v_color.tif

# TODO: stdosl.h implementation doesn't support xyY, an oversight?
#testshade -t 1 -g 64 64 -param colorspace xyY -od uint8 -o Cout sout_transformc_xyY_v_color.tif test_transformc_u_space_v_color
#testshade -t 1 --batched -g 64 64 -param colorspace xyY -od uint8 -o Cout bout_transformc_xyY_v_color.tif test_transformc_u_space_v_color
#idiff sout_transformc_xyY_v_color.tif bout_transformc_xyY_v_color.tif


# transformc varying colorspace u color includes masking
oslc test_transformc_v_space_u_color.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transformc_v_space_u_color.tif test_transformc_v_space_u_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transformc_v_space_u_color.tif test_transformc_v_space_u_color
idiff sout_transformc_v_space_u_color.tif bout_transformc_v_space_u_color.tif

# transformc varying colorspace v color includes masking
oslc test_transformc_v_space_v_color.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transformc_v_space_v_color.tif test_transformc_v_space_v_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transformc_v_space_v_color.tif test_transformc_v_space_v_color
idiff sout_transformc_v_space_v_color.tif bout_transformc_v_space_v_color.tif



# transformc u fromspace u topace u color includes masking (All to rgb)
oslc test_transformc_u_space_u_space_u_color.osl
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace rgb -od uint8 -o Cout sout_transformc_rgb_rgb_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace rgb -od uint8 -o Cout bout_transformc_rgb_rgb_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_rgb_rgb_u_color.tif bout_transformc_rgb_rgb_u_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace rgb -od uint8 -o Cout sout_transformc_RGB_rgb_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace rgb -od uint8 -o Cout bout_transformc_RGB_rgb_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_RGB_rgb_u_color.tif bout_transformc_RGB_rgb_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace rgb -od uint8 -o Cout sout_transformc_hsv_rgb_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace rgb -od uint8 -o Cout bout_transformc_hsv_rgb_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsv_rgb_u_color.tif bout_transformc_hsv_rgb_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace rgb -od uint8 -o Cout sout_transformc_hsl_rgb_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace rgb -od uint8 -o Cout bout_transformc_hsl_rgb_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsl_rgb_u_color.tif bout_transformc_hsl_rgb_u_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace rgb -od uint8 -o Cout sout_transformc_YIQ_rgb_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace rgb -od uint8 -o Cout bout_transformc_YIQ_rgb_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_YIQ_rgb_u_color.tif bout_transformc_YIQ_rgb_u_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace rgb -od uint8 -o Cout sout_transformc_XYZ_rgb_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace rgb -od uint8 -o Cout bout_transformc_XYZ_rgb_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_XYZ_rgb_u_color.tif bout_transformc_XYZ_rgb_u_color.tif


# transformc u fromspace u topace u color includes masking (All to RGB)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace RGB -od uint8 -o Cout sout_transformc_rgb_RGB_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace RGB -od uint8 -o Cout bout_transformc_rgb_RGB_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_rgb_RGB_u_color.tif bout_transformc_rgb_RGB_u_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace RGB -od uint8 -o Cout sout_transformc_RGB_RGB_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace RGB -od uint8 -o Cout bout_transformc_RGB_RGB_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_RGB_RGB_u_color.tif bout_transformc_RGB_RGB_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace RGB -od uint8 -o Cout sout_transformc_hsv_RGB_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace RGB -od uint8 -o Cout bout_transformc_hsv_RGB_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsv_RGB_u_color.tif bout_transformc_hsv_RGB_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace RGB -od uint8 -o Cout sout_transformc_hsl_RGB_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace RGB -od uint8 -o Cout bout_transformc_hsl_RGB_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsl_RGB_u_color.tif bout_transformc_hsl_RGB_u_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace RGB -od uint8 -o Cout sout_transformc_YIQ_RGB_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace RGB -od uint8 -o Cout bout_transformc_YIQ_RGB_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_YIQ_RGB_u_color.tif bout_transformc_YIQ_RGB_u_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace RGB -od uint8 -o Cout sout_transformc_XYZ_RGB_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace RGB -od uint8 -o Cout bout_transformc_XYZ_RGB_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_XYZ_RGB_u_color.tif bout_transformc_XYZ_RGB_u_color.tif



# transformc u fromspace u topace u color includes masking (All to hsv)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace hsv -od uint8 -o Cout sout_transformc_rgb_hsv_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace hsv -od uint8 -o Cout bout_transformc_rgb_hsv_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_rgb_hsv_u_color.tif bout_transformc_rgb_hsv_u_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace hsv -od uint8 -o Cout sout_transformc_RGB_hsv_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace hsv -od uint8 -o Cout bout_transformc_RGB_hsv_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_RGB_hsv_u_color.tif bout_transformc_RGB_hsv_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace hsv -od uint8 -o Cout sout_transformc_hsv_hsv_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace hsv -od uint8 -o Cout bout_transformc_hsv_hsv_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsv_hsv_u_color.tif bout_transformc_hsv_hsv_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace hsv -od uint8 -o Cout sout_transformc_hsl_hsv_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace hsv -od uint8 -o Cout bout_transformc_hsl_hsv_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsl_hsv_u_color.tif bout_transformc_hsl_hsv_u_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace hsv -od uint8 -o Cout sout_transformc_YIQ_hsv_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace hsv -od uint8 -o Cout bout_transformc_YIQ_hsv_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_YIQ_hsv_u_color.tif bout_transformc_YIQ_hsv_u_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace hsv -od uint8 -o Cout sout_transformc_XYZ_hsv_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace hsv -od uint8 -o Cout bout_transformc_XYZ_hsv_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_XYZ_hsv_u_color.tif bout_transformc_XYZ_hsv_u_color.tif


# transformc u fromspace u topace u color includes masking (All to hsl)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace hsl -od uint8 -o Cout sout_transformc_rgb_hsl_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace hsl -od uint8 -o Cout bout_transformc_rgb_hsl_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_rgb_hsl_u_color.tif bout_transformc_rgb_hsl_u_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace hsl -od uint8 -o Cout sout_transformc_RGB_hsl_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace hsl -od uint8 -o Cout bout_transformc_RGB_hsl_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_RGB_hsl_u_color.tif bout_transformc_RGB_hsl_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace hsl -od uint8 -o Cout sout_transformc_hsv_hsl_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace hsl -od uint8 -o Cout bout_transformc_hsv_hsl_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsv_hsl_u_color.tif bout_transformc_hsv_hsl_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace hsl -od uint8 -o Cout sout_transformc_hsl_hsl_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace hsl -od uint8 -o Cout bout_transformc_hsl_hsl_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsl_hsl_u_color.tif bout_transformc_hsl_hsl_u_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace hsl -od uint8 -o Cout sout_transformc_YIQ_hsl_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace hsl -od uint8 -o Cout bout_transformc_YIQ_hsl_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_YIQ_hsl_u_color.tif bout_transformc_YIQ_hsl_u_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace hsl -od uint8 -o Cout sout_transformc_XYZ_hsl_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace hsl -od uint8 -o Cout bout_transformc_XYZ_hsl_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_XYZ_hsl_u_color.tif bout_transformc_XYZ_hsl_u_color.tif


# transformc u fromspace u topace u color includes masking (All to YIQ)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace YIQ -od uint8 -o Cout sout_transformc_rgb_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace YIQ -od uint8 -o Cout bout_transformc_rgb_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_rgb_YIQ_u_color.tif bout_transformc_rgb_YIQ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace YIQ -od uint8 -o Cout sout_transformc_RGB_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace YIQ -od uint8 -o Cout bout_transformc_RGB_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_RGB_YIQ_u_color.tif bout_transformc_RGB_YIQ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace YIQ -od uint8 -o Cout sout_transformc_hsv_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace YIQ -od uint8 -o Cout bout_transformc_hsv_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsv_YIQ_u_color.tif bout_transformc_hsv_YIQ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace YIQ -od uint8 -o Cout sout_transformc_hsl_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace YIQ -od uint8 -o Cout bout_transformc_hsl_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsl_YIQ_u_color.tif bout_transformc_hsl_YIQ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace YIQ -od uint8 -o Cout sout_transformc_YIQ_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace YIQ -od uint8 -o Cout bout_transformc_YIQ_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_YIQ_YIQ_u_color.tif bout_transformc_YIQ_YIQ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace YIQ -od uint8 -o Cout sout_transformc_XYZ_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace YIQ -od uint8 -o Cout bout_transformc_XYZ_YIQ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_XYZ_YIQ_u_color.tif bout_transformc_XYZ_YIQ_u_color.tif


# transformc u fromspace u topace u color includes masking (All to XYZ)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace XYZ -od uint8 -o Cout sout_transformc_rgb_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace XYZ -od uint8 -o Cout bout_transformc_rgb_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_rgb_XYZ_u_color.tif bout_transformc_rgb_XYZ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace XYZ -od uint8 -o Cout sout_transformc_RGB_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace XYZ -od uint8 -o Cout bout_transformc_RGB_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_RGB_XYZ_u_color.tif bout_transformc_RGB_XYZ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace XYZ -od uint8 -o Cout sout_transformc_hsv_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace XYZ -od uint8 -o Cout bout_transformc_hsv_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsv_XYZ_u_color.tif bout_transformc_hsv_XYZ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace XYZ -od uint8 -o Cout sout_transformc_hsl_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace XYZ -od uint8 -o Cout bout_transformc_hsl_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_hsl_XYZ_u_color.tif bout_transformc_hsl_XYZ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace XYZ -od uint8 -o Cout sout_transformc_YIQ_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace XYZ -od uint8 -o Cout bout_transformc_YIQ_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_YIQ_XYZ_u_color.tif bout_transformc_YIQ_XYZ_u_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace XYZ -od uint8 -o Cout sout_transformc_XYZ_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace XYZ -od uint8 -o Cout bout_transformc_XYZ_XYZ_u_color.tif test_transformc_u_space_u_space_u_color
idiff sout_transformc_XYZ_XYZ_u_color.tif bout_transformc_XYZ_XYZ_u_color.tif







# transformc u fromspace u topace v color includes masking (All to rgb)
oslc test_transformc_u_space_u_space_v_color.osl
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace rgb -od uint8 -o Cout sout_transformc_rgb_rgb_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace rgb -od uint8 -o Cout bout_transformc_rgb_rgb_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_rgb_rgb_v_color.tif bout_transformc_rgb_rgb_v_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace rgb -od uint8 -o Cout sout_transformc_RGB_rgb_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace rgb -od uint8 -o Cout bout_transformc_RGB_rgb_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_RGB_rgb_v_color.tif bout_transformc_RGB_rgb_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace rgb -od uint8 -o Cout sout_transformc_hsv_rgb_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace rgb -od uint8 -o Cout bout_transformc_hsv_rgb_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsv_rgb_v_color.tif bout_transformc_hsv_rgb_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace rgb -od uint8 -o Cout sout_transformc_hsl_rgb_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace rgb -od uint8 -o Cout bout_transformc_hsl_rgb_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsl_rgb_v_color.tif bout_transformc_hsl_rgb_v_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace rgb -od uint8 -o Cout sout_transformc_YIQ_rgb_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace rgb -od uint8 -o Cout bout_transformc_YIQ_rgb_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_YIQ_rgb_v_color.tif bout_transformc_YIQ_rgb_v_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace rgb -od uint8 -o Cout sout_transformc_XYZ_rgb_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace rgb -od uint8 -o Cout bout_transformc_XYZ_rgb_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_XYZ_rgb_v_color.tif bout_transformc_XYZ_rgb_v_color.tif


# transformc u fromspace u topace v color includes masking (All to RGB)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace RGB -od uint8 -o Cout sout_transformc_rgb_RGB_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace RGB -od uint8 -o Cout bout_transformc_rgb_RGB_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_rgb_RGB_v_color.tif bout_transformc_rgb_RGB_v_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace RGB -od uint8 -o Cout sout_transformc_RGB_RGB_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace RGB -od uint8 -o Cout bout_transformc_RGB_RGB_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_RGB_RGB_v_color.tif bout_transformc_RGB_RGB_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace RGB -od uint8 -o Cout sout_transformc_hsv_RGB_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace RGB -od uint8 -o Cout bout_transformc_hsv_RGB_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsv_RGB_v_color.tif bout_transformc_hsv_RGB_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace RGB -od uint8 -o Cout sout_transformc_hsl_RGB_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace RGB -od uint8 -o Cout bout_transformc_hsl_RGB_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsl_RGB_v_color.tif bout_transformc_hsl_RGB_v_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace RGB -od uint8 -o Cout sout_transformc_YIQ_RGB_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace RGB -od uint8 -o Cout bout_transformc_YIQ_RGB_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_YIQ_RGB_v_color.tif bout_transformc_YIQ_RGB_v_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace RGB -od uint8 -o Cout sout_transformc_XYZ_RGB_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace RGB -od uint8 -o Cout bout_transformc_XYZ_RGB_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_XYZ_RGB_v_color.tif bout_transformc_XYZ_RGB_v_color.tif



# transformc u fromspace u topace v color includes masking (All to hsv)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace hsv -od uint8 -o Cout sout_transformc_rgb_hsv_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace hsv -od uint8 -o Cout bout_transformc_rgb_hsv_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_rgb_hsv_v_color.tif bout_transformc_rgb_hsv_v_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace hsv -od uint8 -o Cout sout_transformc_RGB_hsv_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace hsv -od uint8 -o Cout bout_transformc_RGB_hsv_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_RGB_hsv_v_color.tif bout_transformc_RGB_hsv_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace hsv -od uint8 -o Cout sout_transformc_hsv_hsv_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace hsv -od uint8 -o Cout bout_transformc_hsv_hsv_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsv_hsv_v_color.tif bout_transformc_hsv_hsv_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace hsv -od uint8 -o Cout sout_transformc_hsl_hsv_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace hsv -od uint8 -o Cout bout_transformc_hsl_hsv_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsl_hsv_v_color.tif bout_transformc_hsl_hsv_v_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace hsv -od uint8 -o Cout sout_transformc_YIQ_hsv_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace hsv -od uint8 -o Cout bout_transformc_YIQ_hsv_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_YIQ_hsv_v_color.tif bout_transformc_YIQ_hsv_v_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace hsv -od uint8 -o Cout sout_transformc_XYZ_hsv_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace hsv -od uint8 -o Cout bout_transformc_XYZ_hsv_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_XYZ_hsv_v_color.tif bout_transformc_XYZ_hsv_v_color.tif


# transformc u fromspace u topace v color includes masking (All to hsl)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace hsl -od uint8 -o Cout sout_transformc_rgb_hsl_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace hsl -od uint8 -o Cout bout_transformc_rgb_hsl_v_color.tif test_transformc_u_space_u_space_v_color
idiff -fail 0.004 sout_transformc_rgb_hsl_v_color.tif bout_transformc_rgb_hsl_v_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace hsl -od uint8 -o Cout sout_transformc_RGB_hsl_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace hsl -od uint8 -o Cout bout_transformc_RGB_hsl_v_color.tif test_transformc_u_space_u_space_v_color
idiff -fail 0.004 sout_transformc_RGB_hsl_v_color.tif bout_transformc_RGB_hsl_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace hsl -od uint8 -o Cout sout_transformc_hsv_hsl_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace hsl -od uint8 -o Cout bout_transformc_hsv_hsl_v_color.tif test_transformc_u_space_u_space_v_color
idiff -fail 0.004 sout_transformc_hsv_hsl_v_color.tif bout_transformc_hsv_hsl_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace hsl -od uint8 -o Cout sout_transformc_hsl_hsl_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace hsl -od uint8 -o Cout bout_transformc_hsl_hsl_v_color.tif test_transformc_u_space_u_space_v_color
idiff -fail 0.004 sout_transformc_hsl_hsl_v_color.tif bout_transformc_hsl_hsl_v_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace hsl -od uint8 -o Cout sout_transformc_YIQ_hsl_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace hsl -od uint8 -o Cout bout_transformc_YIQ_hsl_v_color.tif test_transformc_u_space_u_space_v_color
idiff -fail 0.004 sout_transformc_YIQ_hsl_v_color.tif bout_transformc_YIQ_hsl_v_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace hsl -od uint8 -o Cout sout_transformc_XYZ_hsl_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace hsl -od uint8 -o Cout bout_transformc_XYZ_hsl_v_color.tif test_transformc_u_space_u_space_v_color
idiff -fail 0.004 sout_transformc_XYZ_hsl_v_color.tif bout_transformc_XYZ_hsl_v_color.tif


# transformc u fromspace u topace v color includes masking (All to YIQ)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace YIQ -od uint8 -o Cout sout_transformc_rgb_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace YIQ -od uint8 -o Cout bout_transformc_rgb_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_rgb_YIQ_v_color.tif bout_transformc_rgb_YIQ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace YIQ -od uint8 -o Cout sout_transformc_RGB_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace YIQ -od uint8 -o Cout bout_transformc_RGB_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_RGB_YIQ_v_color.tif bout_transformc_RGB_YIQ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace YIQ -od uint8 -o Cout sout_transformc_hsv_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace YIQ -od uint8 -o Cout bout_transformc_hsv_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsv_YIQ_v_color.tif bout_transformc_hsv_YIQ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace YIQ -od uint8 -o Cout sout_transformc_hsl_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace YIQ -od uint8 -o Cout bout_transformc_hsl_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsl_YIQ_v_color.tif bout_transformc_hsl_YIQ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace YIQ -od uint8 -o Cout sout_transformc_YIQ_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace YIQ -od uint8 -o Cout bout_transformc_YIQ_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_YIQ_YIQ_v_color.tif bout_transformc_YIQ_YIQ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace YIQ -od uint8 -o Cout sout_transformc_XYZ_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace YIQ -od uint8 -o Cout bout_transformc_XYZ_YIQ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_XYZ_YIQ_v_color.tif bout_transformc_XYZ_YIQ_v_color.tif


# transformc u fromspace u topace v color includes masking (All to XYZ)
testshade -t 1 -g 64 64 -param fromspace rgb -param tospace XYZ -od uint8 -o Cout sout_transformc_rgb_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -param tospace XYZ -od uint8 -o Cout bout_transformc_rgb_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_rgb_XYZ_v_color.tif bout_transformc_rgb_XYZ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -param tospace XYZ -od uint8 -o Cout sout_transformc_RGB_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -param tospace XYZ -od uint8 -o Cout bout_transformc_RGB_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_RGB_XYZ_v_color.tif bout_transformc_RGB_XYZ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -param tospace XYZ -od uint8 -o Cout sout_transformc_hsv_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -param tospace XYZ -od uint8 -o Cout bout_transformc_hsv_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsv_XYZ_v_color.tif bout_transformc_hsv_XYZ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -param tospace XYZ -od uint8 -o Cout sout_transformc_hsl_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -param tospace XYZ -od uint8 -o Cout bout_transformc_hsl_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_hsl_XYZ_v_color.tif bout_transformc_hsl_XYZ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -param tospace XYZ -od uint8 -o Cout sout_transformc_YIQ_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -param tospace XYZ -od uint8 -o Cout bout_transformc_YIQ_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_YIQ_XYZ_v_color.tif bout_transformc_YIQ_XYZ_v_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -param tospace XYZ -od uint8 -o Cout sout_transformc_XYZ_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -param tospace XYZ -od uint8 -o Cout bout_transformc_XYZ_XYZ_v_color.tif test_transformc_u_space_u_space_v_color
idiff sout_transformc_XYZ_XYZ_v_color.tif bout_transformc_XYZ_XYZ_v_color.tif





# transformc u fromspace v tospace v color includes masking
oslc  test_transformc_u_space_v_space_v_color.osl
testshade -t 1 -g 64 64 -param fromspace rgb -od uint8 -o Cout sout_transformc_rgb_v_space_v_color.tif test_transformc_u_space_v_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace rgb -od uint8 -o Cout bout_transformc_rgb_v_space_v_color.tif test_transformc_u_space_v_space_v_color
idiff sout_transformc_rgb_v_space_v_color.tif bout_transformc_rgb_v_space_v_color.tif

testshade -t 1 -g 64 64 -param fromspace RGB -od uint8 -o Cout sout_transformc_RGB_v_space_v_color.tif test_transformc_u_space_v_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace RGB -od uint8 -o Cout bout_transformc_RGB_v_space_v_color.tif test_transformc_u_space_v_space_v_color
idiff sout_transformc_RGB_v_space_v_color.tif bout_transformc_RGB_v_space_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsv -od uint8 -o Cout sout_transformc_hsv_v_space_v_color.tif test_transformc_u_space_v_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsv -od uint8 -o Cout bout_transformc_hsv_v_space_v_color.tif test_transformc_u_space_v_space_v_color
idiff sout_transformc_hsv_v_space_v_color.tif bout_transformc_hsv_v_space_v_color.tif

testshade -t 1 -g 64 64 -param fromspace hsl -od uint8 -o Cout sout_transformc_hsl_v_space_v_color.tif test_transformc_u_space_v_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace hsl -od uint8 -o Cout bout_transformc_hsl_v_space_v_color.tif test_transformc_u_space_v_space_v_color
idiff -fail 0.004 sout_transformc_hsl_v_space_v_color.tif bout_transformc_hsl_v_space_v_color.tif

testshade -t 1 -g 64 64 -param fromspace YIQ -od uint8 -o Cout sout_transformc_YIQ_v_space_v_color.tif test_transformc_u_space_v_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace YIQ -od uint8 -o Cout bout_transformc_YIQ_v_space_v_color.tif test_transformc_u_space_v_space_v_color
idiff sout_transformc_YIQ_v_space_v_color.tif bout_transformc_YIQ_v_space_v_color.tif

testshade -t 1 -g 64 64 -param fromspace XYZ -od uint8 -o Cout sout_transformc_XYZ_v_space_v_color.tif test_transformc_u_space_v_space_v_color
testshade -t 1 --batched -g 64 64 -param fromspace XYZ -od uint8 -o Cout bout_transformc_XYZ_v_space_v_color.tif test_transformc_u_space_v_space_v_color
idiff sout_transformc_XYZ_v_space_v_color.tif bout_transformc_XYZ_v_space_v_color.tif

# TODO: stdosl.h implementation doesn't support xyY, an oversight?
#testshade -t 1 -g 64 64 -param fromspace xyY -od uint8 -o Cout sout_transformc_xyY_v_space_v_color.tif test_transformc_u_space_v_space_v_color
#testshade -t 1 --batched -g 64 64 -param fromspace xyY -od uint8 -o Cout bout_transformc_xyY_v_space_v_color.tif test_transformc_u_space_v_space_v_color
#idiff sout_transformc_xyY_v_space_v_color.tif bout_transformc_xyY_v_space_v_color.tif




# transformc v fromspace u tospace v color includes masking
oslc  test_transformc_v_space_u_space_v_color.osl
testshade -t 1 -g 64 64 -param tospace rgb -od uint8 -o Cout sout_transformc_v_space_rgb_v_color.tif test_transformc_v_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param tospace rgb -od uint8 -o Cout bout_transformc_v_space_rgb_v_color.tif test_transformc_v_space_u_space_v_color
idiff sout_transformc_v_space_rgb_v_color.tif bout_transformc_v_space_rgb_v_color.tif

testshade -t 1 -g 64 64 -param tospace RGB -od uint8 -o Cout sout_transformc_v_space_RGB_v_color.tif test_transformc_v_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param tospace RGB -od uint8 -o Cout bout_transformc_v_space_RGB_v_color.tif test_transformc_v_space_u_space_v_color
idiff sout_transformc_v_space_RGB_v_color.tif bout_transformc_v_space_RGB_v_color.tif

testshade -t 1 -g 64 64 -param tospace hsv -od uint8 -o Cout sout_transformc_v_space_hsv_v_color.tif test_transformc_v_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param tospace hsv -od uint8 -o Cout bout_transformc_v_space_hsv_v_color.tif test_transformc_v_space_u_space_v_color
idiff sout_transformc_v_space_hsv_v_color.tif bout_transformc_v_space_hsv_v_color.tif

testshade -t 1 -g 64 64 -param tospace hsl -od uint8 -o Cout sout_transformc_v_space_hsl_v_color.tif test_transformc_v_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param tospace hsl -od uint8 -o Cout bout_transformc_v_space_hsl_v_color.tif test_transformc_v_space_u_space_v_color
idiff -fail 0.004 sout_transformc_v_space_hsl_v_color.tif bout_transformc_v_space_hsl_v_color.tif

testshade -t 1 -g 64 64 -param tospace YIQ -od uint8 -o Cout sout_transformc_v_space_YIQ_v_color.tif test_transformc_v_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param tospace YIQ -od uint8 -o Cout bout_transformc_v_space_YIQ_v_color.tif test_transformc_v_space_u_space_v_color
idiff sout_transformc_v_space_YIQ_v_color.tif bout_transformc_v_space_YIQ_v_color.tif

testshade -t 1 -g 64 64 -param tospace XYZ -od uint8 -o Cout sout_transformc_v_space_XYZ_v_color.tif test_transformc_v_space_u_space_v_color
testshade -t 1 --batched -g 64 64 -param tospace XYZ -od uint8 -o Cout bout_transformc_v_space_XYZ_v_color.tif test_transformc_v_space_u_space_v_color
idiff sout_transformc_v_space_XYZ_v_color.tif bout_transformc_v_space_XYZ_v_color.tif

# TODO: stdosl.h implementation doesn't support xyY, an oversight?
#testshade -t 1 -g 64 64 -param tospace xyY -od uint8 -o Cout sout_transformc_v_space_xyY_v_color.tif test_transformc_v_space_u_space_v_color
#testshade -t 1 --batched -g 64 64 -param tospace xyY -od uint8 -o Cout bout_transformc_v_space_xyY_v_color.tif test_transformc_v_space_u_space_v_color
#idiff sout_transformc_v_space_xyY_v_color.tif bout_transformc_v_space_xyY_v_color.tif



# transformc v fromspace v tospace v color includes masking
oslc  test_transformc_v_space_v_space_v_color.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transformc_v_space_v_space_v_color.tif test_transformc_v_space_v_space_v_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transformc_v_space_v_space_v_color.tif test_transformc_v_space_v_space_v_color
idiff -fail 0.004 sout_transformc_v_space_v_space_v_color.tif bout_transformc_v_space_v_space_v_color.tif
