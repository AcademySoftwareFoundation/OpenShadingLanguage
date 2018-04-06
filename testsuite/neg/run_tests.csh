rm *.tif *.oso

# neg u matrix includes masking
oslc test_neg_u_matrix.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_neg_u_matrix.tif test_neg_u_matrix
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_neg_u_matrix.tif test_neg_u_matrix
idiff sout_neg_u_matrix.tif bout_neg_u_matrix.tif

# neg v matrix includes masking
oslc test_neg_v_matrix.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_neg_v_matrix.tif test_neg_v_matrix
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_neg_v_matrix.tif test_neg_v_matrix
idiff sout_neg_v_matrix.tif bout_neg_v_matrix.tif



# neg u int includes masking
oslc test_neg_u_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_neg_u_int.tif test_neg_u_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_neg_u_int.tif test_neg_u_int
idiff sout_neg_u_int.tif bout_neg_u_int.tif

# neg v int includes masking
oslc test_neg_v_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_neg_v_int.tif test_neg_v_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_neg_v_int.tif test_neg_v_int
idiff sout_neg_v_int.tif bout_neg_v_int.tif


# neg u float includes masking
oslc test_neg_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_neg_u_float.tif test_neg_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_neg_u_float.tif test_neg_u_float
idiff sout_neg_u_float.tif bout_neg_u_float.tif

# neg v float includes masking
oslc test_neg_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_neg_v_float.tif test_neg_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_neg_v_float.tif test_neg_v_float
idiff sout_neg_v_float.tif bout_neg_v_float.tif

# neg v dfloat includes masking
oslc test_neg_v_dfloat.osl
testshade -t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout sout_neg_v_dfloat.tif test_neg_v_dfloat
testshade -t 1 --batched -g 64 64 --vary_pdxdy -od uint8 -o Cout bout_neg_v_dfloat.tif test_neg_v_dfloat
idiff sout_neg_v_dfloat.tif bout_neg_v_dfloat.tif


# neg u color includes masking
oslc test_neg_u_color.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_neg_u_color.tif test_neg_u_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_neg_u_color.tif test_neg_u_color
idiff sout_neg_u_color.tif bout_neg_u_color.tif

# neg v color includes masking
oslc test_neg_v_color.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_neg_v_color.tif test_neg_v_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_neg_v_color.tif test_neg_v_color
idiff sout_neg_v_color.tif bout_neg_v_color.tif

# neg v dcolor includes masking
oslc test_neg_v_dcolor.osl
testshade -t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout sout_neg_v_dcolor.tif test_neg_v_dcolor
testshade -t 1 --batched -g 64 64 --vary_pdxdy -od uint8 -o Cout bout_neg_v_dcolor.tif test_neg_v_dcolor
idiff sout_neg_v_dcolor.tif bout_neg_v_dcolor.tif

