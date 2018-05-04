# matrix u fromspace u float includes masking
testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout sout_matrix_$1_fromspace_u_float.tif test_matrix_u_fromspace_u_float
testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout bout_matrix_$1_fromspace_u_float.tif test_matrix_u_fromspace_u_float
idiff sout_matrix_$1_fromspace_u_float.tif bout_matrix_$1_fromspace_u_float.tif

# matrix u fromspace v float includes masking
testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout sout_matrix_$1_fromspace_v_float.tif test_matrix_u_fromspace_v_float
testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout bout_matrix_$1_fromspace_v_float.tif test_matrix_u_fromspace_v_float
idiff sout_matrix_$1_fromspace_v_float.tif bout_matrix_$1_fromspace_v_float.tif


