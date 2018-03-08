rm *.tif *.oso


# matrix / matrix
oslc test_u_matrix_div_u_matrix.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_matrix_div_u_matrix.tif test_u_matrix_div_u_matrix
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_matrix_div_u_matrix.tif test_u_matrix_div_u_matrix
idiff sout_u_matrix_div_u_matrix.tif bout_u_matrix_div_u_matrix.tif

oslc test_u_matrix_div_v_matrix.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_matrix_div_v_matrix.tif test_u_matrix_div_v_matrix
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_matrix_div_v_matrix.tif test_u_matrix_div_v_matrix
idiff sout_u_matrix_div_v_matrix.tif bout_u_matrix_div_v_matrix.tif

oslc test_v_matrix_div_u_matrix.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_matrix_div_u_matrix.tif test_v_matrix_div_u_matrix
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_matrix_div_u_matrix.tif test_v_matrix_div_u_matrix
idiff sout_v_matrix_div_u_matrix.tif bout_v_matrix_div_u_matrix.tif

oslc test_v_matrix_div_v_matrix.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_matrix_div_v_matrix.tif test_v_matrix_div_v_matrix
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_matrix_div_v_matrix.tif test_v_matrix_div_v_matrix
idiff sout_v_matrix_div_v_matrix.tif bout_v_matrix_div_v_matrix.tif


# matrix / matrix MASKED
oslc test_u_matrix_div_u_matrix_masked.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_matrix_div_u_matrix_masked.tif test_u_matrix_div_u_matrix_masked
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_matrix_div_u_matrix_masked.tif test_u_matrix_div_u_matrix_masked
idiff sout_u_matrix_div_u_matrix_masked.tif bout_u_matrix_div_u_matrix_masked.tif

oslc test_u_matrix_div_v_matrix_masked.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_matrix_div_v_matrix_masked.tif test_u_matrix_div_v_matrix_masked
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_matrix_div_v_matrix_masked.tif test_u_matrix_div_v_matrix_masked
idiff sout_u_matrix_div_v_matrix_masked.tif bout_u_matrix_div_v_matrix_masked.tif

oslc test_v_matrix_div_u_matrix_masked.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_matrix_div_u_matrix_masked.tif test_v_matrix_div_u_matrix_masked
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_matrix_div_u_matrix_masked.tif test_v_matrix_div_u_matrix_masked
idiff sout_v_matrix_div_u_matrix_masked.tif bout_v_matrix_div_u_matrix_masked.tif

oslc test_v_matrix_div_v_matrix_masked.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_matrix_div_v_matrix_masked.tif test_v_matrix_div_v_matrix_masked
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_matrix_div_v_matrix_masked.tif test_v_matrix_div_v_matrix_masked
idiff sout_v_matrix_div_v_matrix_masked.tif bout_v_matrix_div_v_matrix_masked.tif

