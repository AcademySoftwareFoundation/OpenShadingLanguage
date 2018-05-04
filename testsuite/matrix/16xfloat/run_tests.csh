rm *.tif *.oso

# matrix 16x u float includes masking
oslc test_matrix_16x_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_matrix_16x_u_float.tif test_matrix_16x_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_matrix_16x_u_float.tif test_matrix_16x_u_float
idiff sout_matrix_16x_u_float.tif bout_matrix_16x_u_float.tif

# matrix 16x v float includes masking
oslc test_matrix_16x_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_matrix_16x_v_float.tif test_matrix_16x_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_matrix_16x_v_float.tif test_matrix_16x_v_float
idiff sout_matrix_16x_v_float.tif bout_matrix_16x_v_float.tif


