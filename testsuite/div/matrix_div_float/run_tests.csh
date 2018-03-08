rm *.tif *.oso

# matrix / float
oslc test_u_matrix_div_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_matrix_div_u_float.tif test_u_matrix_div_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_matrix_div_u_float.tif test_u_matrix_div_u_float
idiff sout_u_matrix_div_u_float.tif bout_u_matrix_div_u_float.tif

oslc test_u_matrix_div_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_matrix_div_v_float.tif test_u_matrix_div_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_matrix_div_v_float.tif test_u_matrix_div_v_float
idiff sout_u_matrix_div_v_float.tif bout_u_matrix_div_v_float.tif


oslc test_v_matrix_div_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_matrix_div_u_float.tif test_v_matrix_div_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_matrix_div_u_float.tif test_v_matrix_div_u_float
idiff sout_v_matrix_div_u_float.tif bout_v_matrix_div_u_float.tif

oslc test_v_matrix_div_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_matrix_div_v_float.tif test_v_matrix_div_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_matrix_div_v_float.tif test_v_matrix_div_v_float
idiff sout_v_matrix_div_v_float.tif bout_v_matrix_div_v_float.tif



# matrix / float MASKED
oslc test_u_matrix_div_u_float_masked.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_matrix_div_u_float_masked.tif test_u_matrix_div_u_float_masked
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_matrix_div_u_float_masked.tif test_u_matrix_div_u_float_masked
idiff sout_u_matrix_div_u_float_masked.tif bout_u_matrix_div_u_float_masked.tif

oslc test_u_matrix_div_v_float_masked.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_matrix_div_v_float_masked.tif test_u_matrix_div_v_float_masked
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_matrix_div_v_float_masked.tif test_u_matrix_div_v_float_masked
idiff sout_u_matrix_div_v_float_masked.tif bout_u_matrix_div_v_float_masked.tif


oslc test_v_matrix_div_u_float_masked.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_matrix_div_u_float_masked.tif test_v_matrix_div_u_float_masked
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_matrix_div_u_float_masked.tif test_v_matrix_div_u_float_masked
idiff sout_v_matrix_div_u_float_masked.tif bout_v_matrix_div_u_float_masked.tif


oslc test_v_matrix_div_v_float_masked.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_matrix_div_v_float_masked.tif test_v_matrix_div_v_float_masked
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_matrix_div_v_float_masked.tif test_v_matrix_div_v_float_masked
idiff sout_v_matrix_div_v_float_masked.tif bout_v_matrix_div_v_float_masked.tif

