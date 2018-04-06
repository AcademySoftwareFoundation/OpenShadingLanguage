rm *.tif *.oso

# abs u float includes masking
oslc test_abs_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_abs_u_float.tif test_abs_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_abs_u_float.tif test_abs_u_float
idiff sout_abs_u_float.tif bout_abs_u_float.tif

# abs v float includes masking
oslc test_abs_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_abs_v_float.tif test_abs_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_abs_v_float.tif test_abs_v_float
idiff sout_abs_v_float.tif bout_abs_v_float.tif

# abs v dfloat includes masking
oslc test_abs_v_dfloat.osl
testshade -t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout sout_abs_v_dfloat.tif test_abs_v_dfloat
testshade -t 1 --batched -g 64 64 --vary_pdxdy -od uint8 -o Cout bout_abs_v_dfloat.tif test_abs_v_dfloat
idiff sout_abs_v_dfloat.tif bout_abs_v_dfloat.tif


# abs u color includes masking
oslc test_abs_u_color.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_abs_u_color.tif test_abs_u_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_abs_u_color.tif test_abs_u_color
idiff sout_abs_u_color.tif bout_abs_u_color.tif

# abs v color includes masking
oslc test_abs_v_color.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_abs_v_color.tif test_abs_v_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_abs_v_color.tif test_abs_v_color
idiff sout_abs_v_color.tif bout_abs_v_color.tif

# abs v dcolor includes masking
oslc test_abs_v_dcolor.osl
testshade -t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout sout_abs_v_dcolor.tif test_abs_v_dcolor
testshade -t 1 --batched -g 64 64 --vary_pdxdy -od uint8 -o Cout bout_abs_v_dcolor.tif test_abs_v_dcolor
idiff sout_abs_v_dcolor.tif bout_abs_v_dcolor.tif

