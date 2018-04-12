rm *.tif *.oso

# transform u matrix u point includes masking
oslc test_transform_u_matrix_u_point.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_u_matrix_u_point.tif test_transform_u_matrix_u_point
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_u_matrix_u_point.tif test_transform_u_matrix_u_point
idiff sout_transform_u_matrix_u_point.tif bout_transform_u_matrix_u_point.tif


# transform u matrix v point includes masking
oslc test_transform_u_matrix_v_point.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_u_matrix_v_point.tif test_transform_u_matrix_v_point
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_u_matrix_v_point.tif test_transform_u_matrix_v_point
idiff sout_transform_u_matrix_v_point.tif bout_transform_u_matrix_v_point.tif

# transform u matrix v dual point includes masking
oslc test_transform_u_matrix_v_dpoint.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_transform_u_matrix_v_dpoint.tif test_transform_u_matrix_v_dpoint
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_u_matrix_v_dpoint.tif test_transform_u_matrix_v_dpoint
idiff sout_transform_u_matrix_v_dpoint.tif bout_transform_u_matrix_v_dpoint.tif





# transform v matrix u point includes masking
oslc test_transform_v_matrix_u_point.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_matrix_u_point.tif test_transform_v_matrix_u_point
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_matrix_u_point.tif test_transform_v_matrix_u_point
idiff sout_transform_v_matrix_u_point.tif bout_transform_v_matrix_u_point.tif


# transform v matrix v point includes masking
oslc test_transform_v_matrix_v_point.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_matrix_v_point.tif test_transform_v_matrix_v_point
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_matrix_v_point.tif test_transform_v_matrix_v_point
idiff sout_transform_v_matrix_v_point.tif bout_transform_v_matrix_v_point.tif

# transform v matrix v dual point includes masking
oslc test_transform_v_matrix_v_dpoint.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_matrix_v_dpoint.tif test_transform_v_matrix_v_dpoint
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_matrix_v_dpoint.tif test_transform_v_matrix_v_dpoint
idiff sout_transform_v_matrix_v_dpoint.tif bout_transform_v_matrix_v_dpoint.tif
