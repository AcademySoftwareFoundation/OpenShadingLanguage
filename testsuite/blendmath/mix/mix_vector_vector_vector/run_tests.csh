rm *.tif *.oso

# mix vector, vector, vector includes masking
oslc test_mix_u_vector_u_vector_u_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_u_vector_u_vector_u_vector.tif test_mix_u_vector_u_vector_u_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_u_vector_u_vector_u_vector.tif test_mix_u_vector_u_vector_u_vector
idiff sout_mix_u_vector_u_vector_u_vector.tif bout_mix_u_vector_u_vector_u_vector.tif

oslc test_mix_u_vector_u_vector_v_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_u_vector_u_vector_v_vector.tif test_mix_u_vector_u_vector_v_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_u_vector_u_vector_v_vector.tif test_mix_u_vector_u_vector_v_vector
idiff sout_mix_u_vector_u_vector_v_vector.tif bout_mix_u_vector_u_vector_v_vector.tif

oslc test_mix_u_vector_v_vector_u_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_u_vector_v_vector_u_vector.tif test_mix_u_vector_v_vector_u_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_u_vector_v_vector_u_vector.tif test_mix_u_vector_v_vector_u_vector
idiff sout_mix_u_vector_v_vector_u_vector.tif bout_mix_u_vector_v_vector_u_vector.tif

oslc test_mix_u_vector_v_vector_v_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_u_vector_v_vector_v_vector.tif test_mix_u_vector_v_vector_v_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_u_vector_v_vector_v_vector.tif test_mix_u_vector_v_vector_v_vector
idiff sout_mix_u_vector_v_vector_v_vector.tif bout_mix_u_vector_v_vector_v_vector.tif

oslc test_mix_v_dvector_v_dvector_c_vector.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_dvector_v_dvector_c_vector.tif test_mix_v_dvector_v_dvector_c_vector
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_dvector_v_dvector_c_vector.tif test_mix_v_dvector_v_dvector_c_vector
idiff sout_mix_v_dvector_v_dvector_c_vector.tif bout_mix_v_dvector_v_dvector_c_vector.tif

oslc test_mix_v_dvector_v_dvector_v_dvector.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_dvector_v_dvector_v_dvector.tif test_mix_v_dvector_v_dvector_v_dvector
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_dvector_v_dvector_v_dvector.tif test_mix_v_dvector_v_dvector_v_dvector
idiff sout_mix_v_dvector_v_dvector_v_dvector.tif bout_mix_v_dvector_v_dvector_v_dvector.tif

oslc test_mix_v_vector_u_vector_u_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_vector_u_vector_u_vector.tif test_mix_v_vector_u_vector_u_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_vector_u_vector_u_vector.tif test_mix_v_vector_u_vector_u_vector
idiff sout_mix_v_vector_u_vector_u_vector.tif bout_mix_v_vector_u_vector_u_vector.tif

oslc test_mix_v_vector_u_vector_v_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_vector_u_vector_v_vector.tif test_mix_v_vector_u_vector_v_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_vector_u_vector_v_vector.tif test_mix_v_vector_u_vector_v_vector
idiff -fail 0.004 sout_mix_v_vector_u_vector_v_vector.tif bout_mix_v_vector_u_vector_v_vector.tif

oslc test_mix_v_vector_v_vector_u_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_vector_v_vector_u_vector.tif test_mix_v_vector_v_vector_u_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_vector_v_vector_u_vector.tif test_mix_v_vector_v_vector_u_vector
idiff sout_mix_v_vector_v_vector_u_vector.tif bout_mix_v_vector_v_vector_u_vector.tif

oslc test_mix_v_vector_v_vector_v_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_mix_v_vector_v_vector_v_vector.tif test_mix_v_vector_v_vector_v_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_mix_v_vector_v_vector_v_vector.tif test_mix_v_vector_v_vector_v_vector
idiff sout_mix_v_vector_v_vector_v_vector.tif bout_mix_v_vector_v_vector_v_vector.tif




