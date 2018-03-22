rm *.tif *.oso

# min(vec, vec) (including Masking)
oslc test_min_u_vec_u_vec.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_u_vec_u_vec.tif test_min_u_vec_u_vec
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_u_vec_u_vec.tif test_min_u_vec_u_vec
idiff sout_min_u_vec_u_vec.tif bout_min_u_vec_u_vec.tif

oslc test_min_u_vec_v_vec.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_v_vec_v_vec.tif test_min_u_vec_v_vec
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_v_vec_v_vec.tif test_min_u_vec_v_vec
idiff sout_min_v_vec_v_vec.tif bout_min_v_vec_v_vec.tif

oslc test_min_v_vec_v_vec.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_v_vec_v_vec.tif test_min_v_vec_v_vec
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_v_vec_v_vec.tif test_min_v_vec_v_vec
idiff sout_min_v_vec_v_vec.tif bout_min_v_vec_v_vec.tif

oslc test_min_v_vec_u_vec.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_v_vec_u_vec.tif test_min_v_vec_u_vec
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_v_vec_u_vec.tif test_min_v_vec_u_vec
idiff sout_min_v_vec_u_vec.tif bout_min_v_vec_u_vec.tif

oslc test_min_v_dvec_v_dvec.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_min_v_dvec_v_dvec.tif test_min_v_dvec_v_dvec
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_v_dvec_v_dvec.tif test_min_v_dvec_v_dvec
idiff sout_min_v_dvec_v_dvec.tif bout_min_v_dvec_v_dvec.tif



