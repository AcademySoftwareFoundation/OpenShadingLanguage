rm *.tif *.oso

# transform u matrix u vector includes masking
oslc test_transform_u_matrix_u_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_u_matrix_u_vector.tif test_transform_u_matrix_u_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_u_matrix_u_vector.tif test_transform_u_matrix_u_vector
idiff sout_transform_u_matrix_u_vector.tif bout_transform_u_matrix_u_vector.tif


# transform u matrix v vector includes masking
oslc test_transform_u_matrix_v_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_u_matrix_v_vector.tif test_transform_u_matrix_v_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_u_matrix_v_vector.tif test_transform_u_matrix_v_vector
idiff sout_transform_u_matrix_v_vector.tif bout_transform_u_matrix_v_vector.tif

# transform u matrix v dual vector includes masking
oslc test_transform_u_matrix_v_dvector.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_transform_u_matrix_v_dvector.tif test_transform_u_matrix_v_dvector
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_u_matrix_v_dvector.tif test_transform_u_matrix_v_dvector
idiff sout_transform_u_matrix_v_dvector.tif bout_transform_u_matrix_v_dvector.tif





# transform v matrix u vector includes masking
oslc test_transform_v_matrix_u_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_matrix_u_vector.tif test_transform_v_matrix_u_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_matrix_u_vector.tif test_transform_v_matrix_u_vector
idiff sout_transform_v_matrix_u_vector.tif bout_transform_v_matrix_u_vector.tif


# transform v matrix v vector includes masking
oslc test_transform_v_matrix_v_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_matrix_v_vector.tif test_transform_v_matrix_v_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_matrix_v_vector.tif test_transform_v_matrix_v_vector
idiff sout_transform_v_matrix_v_vector.tif bout_transform_v_matrix_v_vector.tif

# transform v matrix v dual vector includes masking
oslc test_transform_v_matrix_v_dvector.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_matrix_v_dvector.tif test_transform_v_matrix_v_dvector
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_matrix_v_dvector.tif test_transform_v_matrix_v_dvector
idiff sout_transform_v_matrix_v_dvector.tif bout_transform_v_matrix_v_dvector.tif
