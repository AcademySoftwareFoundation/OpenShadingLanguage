rm *.tif *.oso

# transform u matrix u normal includes masking
oslc test_transform_u_matrix_u_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_u_matrix_u_normal.tif test_transform_u_matrix_u_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_u_matrix_u_normal.tif test_transform_u_matrix_u_normal
idiff sout_transform_u_matrix_u_normal.tif bout_transform_u_matrix_u_normal.tif


# transform u matrix v normal includes masking
oslc test_transform_u_matrix_v_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_u_matrix_v_normal.tif test_transform_u_matrix_v_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_u_matrix_v_normal.tif test_transform_u_matrix_v_normal
idiff sout_transform_u_matrix_v_normal.tif bout_transform_u_matrix_v_normal.tif

# transform u matrix v dual normal includes masking
oslc test_transform_u_matrix_v_dnormal.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_transform_u_matrix_v_dnormal.tif test_transform_u_matrix_v_dnormal
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_u_matrix_v_dnormal.tif test_transform_u_matrix_v_dnormal
idiff sout_transform_u_matrix_v_dnormal.tif bout_transform_u_matrix_v_dnormal.tif





# transform v matrix u normal includes masking
oslc test_transform_v_matrix_u_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_matrix_u_normal.tif test_transform_v_matrix_u_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_matrix_u_normal.tif test_transform_v_matrix_u_normal
idiff sout_transform_v_matrix_u_normal.tif bout_transform_v_matrix_u_normal.tif


# transform v matrix v normal includes masking
oslc test_transform_v_matrix_v_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_matrix_v_normal.tif test_transform_v_matrix_v_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_matrix_v_normal.tif test_transform_v_matrix_v_normal
idiff sout_transform_v_matrix_v_normal.tif bout_transform_v_matrix_v_normal.tif

# transform v matrix v dual normal includes masking
oslc test_transform_v_matrix_v_dnormal.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_matrix_v_dnormal.tif test_transform_v_matrix_v_dnormal
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_matrix_v_dnormal.tif test_transform_v_matrix_v_dnormal
idiff sout_transform_v_matrix_v_dnormal.tif bout_transform_v_matrix_v_dnormal.tif
