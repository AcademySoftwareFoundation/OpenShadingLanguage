oslc test_varying_index_float.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_float.tif test_varying_index_float
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_float.tif test_varying_index_float
idiff sout_varying_index_float.tif bout_varying_index_float.tif 

oslc test_varying_index_int.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_int.tif test_varying_index_int
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_int.tif test_varying_index_int
idiff sout_varying_index_int.tif bout_varying_index_int.tif

oslc test_varying_index_string.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_string.tif test_varying_index_string
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_string.tif test_varying_index_string
idiff sout_varying_index_string.tif bout_varying_index_string.tif

oslc test_varying_index_matrix.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_matrix.tif test_varying_index_matrix
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_matrix.tif test_varying_index_matrix
idiff sout_varying_index_matrix.tif bout_varying_index_matrix.tif


oslc test_varying_index_color.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_color.tif test_varying_index_color
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_color.tif test_varying_index_color
idiff sout_varying_index_color.tif bout_varying_index_color.tif 

oslc test_varying_index_point.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_point.tif test_varying_index_point
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_point.tif test_varying_index_point
idiff sout_varying_index_point.tif bout_varying_index_point.tif 

oslc test_varying_index_vector.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_vector.tif test_varying_index_vector
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_vector.tif test_varying_index_vector
idiff sout_varying_index_vector.tif bout_varying_index_vector.tif 

oslc test_varying_index_normal.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_normal.tif test_varying_index_normal
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_normal.tif test_varying_index_normal
idiff sout_varying_index_normal.tif bout_varying_index_normal.tif 


oslc test_varying_out_of_bounds_index_int.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_out_of_bounds_index_int.tif test_varying_out_of_bounds_index_int
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_out_of_bounds_index_int.tif test_varying_out_of_bounds_index_int
idiff sout_varying_out_of_bounds_index_int.tif bout_varying_out_of_bounds_index_int.tif

oslc test_varying_out_of_bounds_index_float.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_out_of_bounds_index_float.tif test_varying_out_of_bounds_index_float
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_out_of_bounds_index_float.tif test_varying_out_of_bounds_index_float
idiff sout_varying_out_of_bounds_index_float.tif bout_varying_out_of_bounds_index_float.tif

oslc test_varying_out_of_bounds_index_string.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_out_of_bounds_index_string.tif test_varying_out_of_bounds_index_string
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_out_of_bounds_index_string.tif test_varying_out_of_bounds_index_string
idiff sout_varying_out_of_bounds_index_string.tif bout_varying_out_of_bounds_index_string.tif


oslc test_varying_index_ray.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_ray.tif test_varying_index_ray
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_ray.tif test_varying_index_ray
idiff sout_varying_index_ray.tif bout_varying_index_ray.tif 

oslc test_varying_index_cube.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_cube.tif test_varying_index_cube
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_cube.tif test_varying_index_cube
idiff sout_varying_index_cube.tif bout_varying_index_cube.tif 





