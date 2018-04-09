oslc test_arraycopy_u_color.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_u_color.tif test_arraycopy_u_color
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_u_color.tif test_arraycopy_u_color
idiff sout_arraycopy_u_color.tif bout_arraycopy_u_color.tif 

oslc test_arraycopy_v_color.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_v_color.tif test_arraycopy_v_color
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_v_color.tif test_arraycopy_v_color
idiff sout_arraycopy_v_color.tif bout_arraycopy_v_color.tif 




oslc test_arraycopy_v_matrix.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_v_matrix.tif test_arraycopy_v_matrix
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_v_matrix.tif test_arraycopy_v_matrix
idiff sout_arraycopy_v_matrix.tif bout_arraycopy_v_matrix.tif 

oslc test_arraycopy_u_matrix.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_u_matrix.tif test_arraycopy_u_matrix
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_u_matrix.tif test_arraycopy_u_matrix
idiff sout_arraycopy_u_matrix.tif bout_arraycopy_u_matrix.tif 


oslc test_arraycopy_uv_matrix.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_uv_matrix.tif test_arraycopy_uv_matrix
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_uv_matrix.tif test_arraycopy_uv_matrix
idiff sout_arraycopy_uv_matrix.tif bout_arraycopy_uv_matrix.tif 

oslc test_arraycopy_vu_matrix.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_vu_matrix.tif test_arraycopy_vu_matrix
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_vu_matrix.tif test_arraycopy_vu_matrix
idiff sout_arraycopy_vu_matrix.tif bout_arraycopy_vu_matrix.tif 


oslc test_arraycopy_u_float.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_u_float.tif test_arraycopy_u_float
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_u_float.tif test_arraycopy_u_float
idiff sout_arraycopy_u_float.tif bout_arraycopy_u_float.tif 

oslc test_arraycopy_v_float.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_v_float.tif test_arraycopy_v_float
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_v_float.tif test_arraycopy_v_float
idiff sout_arraycopy_v_float.tif bout_arraycopy_v_float.tif 

oslc test_arraycopy_u_int.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_u_int.tif test_arraycopy_u_int
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_u_int.tif test_arraycopy_u_int
idiff sout_arraycopy_u_int.tif bout_arraycopy_u_int.tif 

oslc test_arraycopy_v_int.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_v_int.tif test_arraycopy_v_int
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_v_int.tif test_arraycopy_v_int
idiff sout_arraycopy_v_int.tif bout_arraycopy_v_int.tif 

oslc test_arraycopy_u_string.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_u_string.tif test_arraycopy_u_string
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_u_string.tif test_arraycopy_u_string
idiff sout_arraycopy_u_string.tif bout_arraycopy_u_string.tif 

oslc test_arraycopy_v_string.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_arraycopy_v_string.tif test_arraycopy_v_string
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_arraycopy_v_string.tif test_arraycopy_v_string
idiff sout_arraycopy_v_string.tif bout_arraycopy_v_string.tif 

