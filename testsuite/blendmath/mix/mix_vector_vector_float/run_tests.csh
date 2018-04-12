rm *.tif *.oso

# mix vector, vector, float includes masking
oslc test_mix_u_vector_u_vector_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_u_vector_u_vector_u_float.tif test_mix_u_vector_u_vector_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_u_vector_u_vector_u_float.tif test_mix_u_vector_u_vector_u_float
idiff sout_mix_u_vector_u_vector_u_float.tif bout_mix_u_vector_u_vector_u_float.tif

oslc test_mix_u_vector_u_vector_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_u_vector_u_vector_v_float.tif test_mix_u_vector_u_vector_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_u_vector_u_vector_v_float.tif test_mix_u_vector_u_vector_v_float
idiff sout_mix_u_vector_u_vector_v_float.tif bout_mix_u_vector_u_vector_v_float.tif

oslc test_mix_u_vector_v_vector_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_u_vector_v_vector_u_float.tif test_mix_u_vector_v_vector_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_u_vector_v_vector_u_float.tif test_mix_u_vector_v_vector_u_float
idiff sout_mix_u_vector_v_vector_u_float.tif bout_mix_u_vector_v_vector_u_float.tif

oslc test_mix_u_vector_v_vector_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_u_vector_v_vector_v_float.tif test_mix_u_vector_v_vector_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_u_vector_v_vector_v_float.tif test_mix_u_vector_v_vector_v_float
idiff sout_mix_u_vector_v_vector_v_float.tif bout_mix_u_vector_v_vector_v_float.tif

oslc test_mix_v_dvector_v_dvector_c_float.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_dvector_v_dvector_c_float.tif test_mix_v_dvector_v_dvector_c_float
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_dvector_v_dvector_c_float.tif test_mix_v_dvector_v_dvector_c_float
idiff sout_mix_v_dvector_v_dvector_c_float.tif bout_mix_v_dvector_v_dvector_c_float.tif

oslc test_mix_v_dvector_v_dvector_v_dfloat.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_dvector_v_dvector_v_dfloat.tif test_mix_v_dvector_v_dvector_v_dfloat
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_dvector_v_dvector_v_dfloat.tif test_mix_v_dvector_v_dvector_v_dfloat
idiff sout_mix_v_dvector_v_dvector_v_dfloat.tif bout_mix_v_dvector_v_dvector_v_dfloat.tif

oslc test_mix_v_vector_u_vector_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_vector_u_vector_u_float.tif test_mix_v_vector_u_vector_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_vector_u_vector_u_float.tif test_mix_v_vector_u_vector_u_float
idiff sout_mix_v_vector_u_vector_u_float.tif bout_mix_v_vector_u_vector_u_float.tif

oslc test_mix_v_vector_u_vector_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_vector_u_vector_v_float.tif test_mix_v_vector_u_vector_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_vector_u_vector_v_float.tif test_mix_v_vector_u_vector_v_float
idiff sout_mix_v_vector_u_vector_v_float.tif bout_mix_v_vector_u_vector_v_float.tif

oslc test_mix_v_vector_v_vector_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_vector_v_vector_u_float.tif test_mix_v_vector_v_vector_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_vector_v_vector_u_float.tif test_mix_v_vector_v_vector_u_float
idiff sout_mix_v_vector_v_vector_u_float.tif bout_mix_v_vector_v_vector_u_float.tif

oslc test_mix_v_vector_v_vector_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_vector_v_vector_v_float.tif test_mix_v_vector_v_vector_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_vector_v_vector_v_float.tif test_mix_v_vector_v_vector_v_float
idiff sout_mix_v_vector_v_vector_v_float.tif bout_mix_v_vector_v_vector_v_float.tif




