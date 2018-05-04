rm *.tif *.oso

# length u vector includes masking
oslc test_length_u_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_vector_u_vector.tif test_length_u_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_vector_u_vector.tif test_length_u_vector
idiff sout_vector_u_vector.tif bout_vector_u_vector.tif

# length v vector includes masking
oslc test_length_v_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_vector_v_vector.tif test_length_v_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_vector_v_vector.tif test_length_v_vector
idiff sout_vector_v_vector.tif bout_vector_v_vector.tif

# length v derivatives of vector includes masking
oslc test_length_v_dvector.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_vector_v_dvector.tif test_length_v_dvector
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_vector_v_dvector.tif test_length_v_dvector
idiff sout_vector_v_dvector.tif bout_vector_v_dvector.tif



# length u normal includes masking
oslc test_length_u_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normal_u_normal.tif test_length_u_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normal_u_normal.tif test_length_u_normal
idiff sout_normal_u_normal.tif bout_normal_u_normal.tif

# length v normal includes masking
oslc test_length_v_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normal_v_normal.tif test_length_v_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normal_v_normal.tif test_length_v_normal
idiff sout_normal_v_normal.tif bout_normal_v_normal.tif

# length v derivatives of normal includes masking
oslc test_length_v_dnormal.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_normal_v_dnormal.tif test_length_v_dnormal
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normal_v_dnormal.tif test_length_v_dnormal
idiff sout_normal_v_dnormal.tif bout_normal_v_dnormal.tif
