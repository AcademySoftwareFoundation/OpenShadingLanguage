rm *.tif *.oso

oslc test_matrix_u_fromspace_16x_u_float.osl
oslc test_matrix_u_fromspace_16x_v_float.osl

./run_fromspace_tests.csh common
./run_fromspace_tests.csh object
./run_fromspace_tests.csh shader
./run_fromspace_tests.csh world
./run_fromspace_tests.csh camera

# matrix v fromspace 16x u float includes masking
oslc test_matrix_v_fromspace_16x_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_matrix_v_fromspace_16x_u_float.tif test_matrix_v_fromspace_16x_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_matrix_v_fromspace_16x_u_float.tif test_matrix_v_fromspace_16x_u_float
idiff sout_matrix_v_fromspace_16x_u_float.tif bout_matrix_v_fromspace_16x_u_float.tif

# matrix v fromspace 16x v float includes masking
oslc test_matrix_v_fromspace_16x_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_matrix_v_fromspace_16x_v_float.tif test_matrix_v_fromspace_16x_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_matrix_v_fromspace_16x_v_float.tif test_matrix_v_fromspace_16x_v_float
idiff sout_matrix_v_fromspace_16x_v_float.tif bout_matrix_v_fromspace_16x_v_float.tif
