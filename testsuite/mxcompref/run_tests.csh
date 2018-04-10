oslc test_compref_u_matrix_const_index.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_compref_u_matrix_const_index.tif test_compref_u_matrix_const_index
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_compref_u_matrix_const_index.tif test_compref_u_matrix_const_index
idiff sout_compref_u_matrix_const_index.tif bout_compref_u_matrix_const_index.tif 

oslc test_compref_v_matrix_const_index.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_compref_v_matrix_const_index.tif test_compref_v_matrix_const_index
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_compref_v_matrix_const_index.tif test_compref_v_matrix_const_index
idiff sout_compref_v_matrix_const_index.tif bout_compref_v_matrix_const_index.tif 

oslc test_compref_u_matrix_u_index.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_compref_u_matrix_u_index.tif test_compref_u_matrix_u_index
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_compref_u_matrix_u_index.tif test_compref_u_matrix_u_index
idiff sout_compref_u_matrix_u_index.tif bout_compref_u_matrix_u_index.tif 

oslc test_compref_v_matrix_u_index.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_compref_v_matrix_u_index.tif test_compref_v_matrix_u_index
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_compref_v_matrix_u_index.tif test_compref_v_matrix_u_index
idiff sout_compref_v_matrix_u_index.tif bout_compref_v_matrix_u_index.tif 



oslc test_compref_u_matrix_v_index.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_compref_u_matrix_v_index.tif test_compref_u_matrix_v_index
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_compref_u_matrix_v_index.tif test_compref_u_matrix_v_index
idiff sout_compref_u_matrix_v_index.tif bout_compref_u_matrix_v_index.tif 

oslc test_compref_v_matrix_v_index.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_compref_v_matrix_v_index.tif test_compref_v_matrix_v_index
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_compref_v_matrix_v_index.tif test_compref_v_matrix_v_index
idiff sout_compref_v_matrix_v_index.tif bout_compref_v_matrix_v_index.tif 
