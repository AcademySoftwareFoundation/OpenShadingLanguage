rm *.tif *.oso

oslc test_getmatrix_u_fromspace_u_tospace.osl
oslc test_getmatrix_u_fromspace_v_tospace.osl
oslc test_getmatrix_v_fromspace_u_tospace.osl
oslc test_getmatrix_v_fromspace_v_tospace.osl


# matrix v fromspace v tospace includes masking
oslc test_getmatrix_v_fromspace_v_tospace.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_matrix_v_fromspace_v_tospace.tif test_getmatrix_v_fromspace_v_tospace
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_matrix_v_fromspace_v_tospace.tif test_getmatrix_v_fromspace_v_tospace
idiff sout_matrix_v_fromspace_v_tospace.tif bout_matrix_v_fromspace_v_tospace.tif

./run_fromspace_tests.csh common
./run_fromspace_tests.csh object
./run_fromspace_tests.csh shader
./run_fromspace_tests.csh world
./run_fromspace_tests.csh camera


./run_tospace_tests.csh common
./run_tospace_tests.csh object
./run_tospace_tests.csh shader
./run_tospace_tests.csh world
./run_tospace_tests.csh camera


