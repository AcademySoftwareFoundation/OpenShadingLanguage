rm *.tif *.oso

# we will take the liberty of assuming point, vector, and normal all take identical code path

# compassign u index u float includes masking
oslc test_compassign_u_index_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_compassign_u_index_u_float.tif test_compassign_u_index_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_compassign_u_index_u_float.tif test_compassign_u_index_u_float
idiff sout_compassign_u_index_u_float.tif bout_compassign_u_index_u_float.tif

# compassign u index v float includes masking
oslc test_compassign_u_index_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_compassign_u_index_v_float.tif test_compassign_u_index_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_compassign_u_index_v_float.tif test_compassign_u_index_v_float
idiff sout_compassign_u_index_v_float.tif bout_compassign_u_index_v_float.tif

# compassign u index v dual float includes masking
oslc test_compassign_u_index_v_dfloat.osl
testshade -t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout sout_compassign_u_index_v_dfloat.tif test_compassign_u_index_v_dfloat
testshade -t 1 --batched -g 64 64 --vary_pdxdy -od uint8 -o Cout bout_compassign_u_index_v_dfloat.tif test_compassign_u_index_v_dfloat
idiff sout_compassign_u_index_v_dfloat.tif bout_compassign_u_index_v_dfloat.tif



# compassign v index u float includes masking
oslc test_compassign_v_index_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_compassign_v_index_u_float.tif test_compassign_v_index_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_compassign_v_index_u_float.tif test_compassign_v_index_u_float
idiff sout_compassign_v_index_u_float.tif bout_compassign_v_index_u_float.tif

# compassign v index v float includes masking
oslc test_compassign_v_index_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_compassign_v_index_v_float.tif test_compassign_v_index_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_compassign_v_index_v_float.tif test_compassign_v_index_v_float
idiff sout_compassign_v_index_v_float.tif bout_compassign_v_index_v_float.tif

# compassign v index v dual float includes masking
oslc test_compassign_v_index_v_dfloat.osl
testshade -t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout sout_compassign_v_index_v_dfloat.tif test_compassign_v_index_v_dfloat
testshade -t 1 --batched -g 64 64 --vary_pdxdy -od uint8 -o Cout bout_compassign_v_index_v_dfloat.tif test_compassign_v_index_v_dfloat
idiff sout_compassign_v_index_v_dfloat.tif bout_compassign_v_index_v_dfloat.tif
