./run_fromspace_tospace_tests.csh $1 common
./run_fromspace_tospace_tests.csh $1 object
./run_fromspace_tospace_tests.csh $1 shader
./run_fromspace_tospace_tests.csh $1 world
./run_fromspace_tospace_tests.csh $1 camera

# matrix u fromspace v tospace includes masking
testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout sout_matrix_$1_fromspace_v_tospace.tif test_matrix_u_fromspace_v_tospace
testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout bout_matrix_$1_fromspace_v_tospace.tif test_matrix_u_fromspace_v_tospace
idiff sout_matrix_$1_fromspace_v_tospace.tif bout_matrix_$1_fromspace_v_tospace.tif
