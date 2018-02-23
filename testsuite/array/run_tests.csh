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



oslc test_varying_index_varying_float.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_varying_float.tif test_varying_index_varying_float
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_varying_float.tif test_varying_index_varying_float
idiff sout_varying_index_varying_float.tif bout_varying_index_varying_float.tif 

oslc test_varying_index_varying_int.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_varying_int.tif test_varying_index_varying_int
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_varying_int.tif test_varying_index_varying_int
idiff sout_varying_index_varying_int.tif bout_varying_index_varying_int.tif 

oslc test_varying_index_varying_point.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_varying_point.tif test_varying_index_varying_point
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_varying_point.tif test_varying_index_varying_point
idiff sout_varying_index_varying_point.tif bout_varying_index_varying_point.tif 

oslc test_varying_index_varying_normal.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_varying_normal.tif test_varying_index_varying_normal
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_varying_normal.tif test_varying_index_varying_normal
idiff sout_varying_index_varying_normal.tif bout_varying_index_varying_normal.tif 

oslc test_varying_index_varying_vector.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_varying_vector.tif test_varying_index_varying_vector
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_varying_vector.tif test_varying_index_varying_vector
idiff sout_varying_index_varying_vector.tif bout_varying_index_varying_vector.tif 

oslc test_varying_index_varying_color.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_varying_color.tif test_varying_index_varying_color
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_varying_color.tif test_varying_index_varying_color
idiff sout_varying_index_varying_color.tif bout_varying_index_varying_color.tif 

oslc test_varying_index_varying_string.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_varying_string.tif test_varying_index_varying_string
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_varying_string.tif test_varying_index_varying_string
idiff sout_varying_index_varying_string.tif bout_varying_index_varying_string.tif 

oslc test_varying_index_varying_matrix.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_varying_matrix.tif test_varying_index_varying_matrix
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_varying_matrix.tif test_varying_index_varying_matrix
idiff sout_varying_index_varying_matrix.tif bout_varying_index_varying_matrix.tif

oslc test_varying_index_varying_ray.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_varying_index_varying_ray.tif test_varying_index_varying_ray
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_varying_index_varying_ray.tif test_varying_index_varying_ray
idiff sout_varying_index_varying_ray.tif bout_varying_index_varying_ray.tif 



oslc test_uniform_index_varying_float.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_uniform_index_varying_float.tif test_uniform_index_varying_float
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_uniform_index_varying_float.tif test_uniform_index_varying_float
idiff sout_uniform_index_varying_float.tif bout_uniform_index_varying_float.tif 

oslc test_uniform_index_varying_int.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_uniform_index_varying_int.tif test_uniform_index_varying_int
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_uniform_index_varying_int.tif test_uniform_index_varying_int
idiff sout_uniform_index_varying_int.tif bout_uniform_index_varying_int.tif 

oslc test_uniform_index_varying_point.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_uniform_index_varying_point.tif test_uniform_index_varying_point
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_uniform_index_varying_point.tif test_uniform_index_varying_point
idiff sout_uniform_index_varying_point.tif bout_uniform_index_varying_point.tif 

oslc test_uniform_index_varying_normal.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_uniform_index_varying_normal.tif test_uniform_index_varying_normal
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_uniform_index_varying_normal.tif test_uniform_index_varying_normal
idiff sout_uniform_index_varying_normal.tif bout_uniform_index_varying_normal.tif 

oslc test_uniform_index_varying_vector.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_uniform_index_varying_vector.tif test_uniform_index_varying_vector
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_uniform_index_varying_vector.tif test_uniform_index_varying_vector
idiff sout_uniform_index_varying_vector.tif bout_uniform_index_varying_vector.tif 

oslc test_uniform_index_varying_color.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_uniform_index_varying_color.tif test_uniform_index_varying_color
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_uniform_index_varying_color.tif test_uniform_index_varying_color
idiff sout_uniform_index_varying_color.tif bout_uniform_index_varying_color.tif 

oslc test_uniform_index_varying_string.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_uniform_index_varying_string.tif test_uniform_index_varying_string
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_uniform_index_varying_string.tif test_uniform_index_varying_string
idiff sout_uniform_index_varying_string.tif bout_uniform_index_varying_string.tif 

oslc test_uniform_index_varying_matrix.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_uniform_index_varying_matrix.tif test_uniform_index_varying_matrix
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_uniform_index_varying_matrix.tif test_uniform_index_varying_matrix
idiff sout_uniform_index_varying_matrix.tif bout_uniform_index_varying_matrix.tif

oslc test_uniform_index_varying_ray.osl
testshade -t 1 -g 256 256 -od uint8 -o Cout sout_uniform_index_varying_ray.tif test_uniform_index_varying_ray
testshade --batched -t 1 -g 256 256 -od uint8 -o Cout bout_uniform_index_varying_ray.tif test_uniform_index_varying_ray
idiff sout_uniform_index_varying_ray.tif bout_uniform_index_varying_ray.tif 
