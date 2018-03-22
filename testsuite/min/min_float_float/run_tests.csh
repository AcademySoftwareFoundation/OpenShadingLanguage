rm *.tif *.oso

# min(float,float) includes masking
oslc test_min_u_float_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_u_float_u_float.tif test_min_u_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_u_float_u_float.tif test_min_u_float_u_float
idiff sout_min_u_float_u_float.tif bout_min_u_float_u_float.tif

oslc test_min_u_float_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_u_float_v_float.tif test_min_u_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_u_float_v_float.tif test_min_u_float_v_float
idiff sout_min_u_float_v_float.tif bout_min_u_float_v_float.tif

oslc test_min_v_float_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_v_float_u_float.tif test_min_v_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_v_float_u_float.tif test_min_v_float_u_float
idiff sout_min_v_float_u_float.tif bout_min_v_float_u_float.tif

oslc test_min_v_float_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_v_float_v_float.tif test_min_v_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_v_float_v_float.tif test_min_v_float_v_float
idiff sout_min_v_float_v_float.tif bout_min_v_float_v_float.tif



# Derivs includes masking
oslc test_min_v_dfloat_v_dfloat.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_min_v_dfloat_v_dfloat.tif test_min_v_dfloat_v_dfloat
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_v_dfloat_v_dfloat.tif test_min_v_dfloat_v_dfloat
idiff sout_min_v_dfloat_v_dfloat.tif bout_min_v_dfloat_v_dfloat.tif

