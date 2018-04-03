rm *.tif *.oso

# max(vec, vec) (including Masking)
oslc test_max_u_vec_u_vec.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_max_u_vec_u_vec.tif test_max_u_vec_u_vec
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_max_u_vec_u_vec.tif test_max_u_vec_u_vec
idiff sout_max_u_vec_u_vec.tif bout_max_u_vec_u_vec.tif

oslc test_max_u_vec_v_vec.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_max_v_vec_v_vec.tif test_max_u_vec_v_vec
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_max_v_vec_v_vec.tif test_max_u_vec_v_vec
idiff sout_max_v_vec_v_vec.tif bout_max_v_vec_v_vec.tif

oslc test_max_v_vec_v_vec.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_max_v_vec_v_vec.tif test_max_v_vec_v_vec
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_max_v_vec_v_vec.tif test_max_v_vec_v_vec
idiff sout_max_v_vec_v_vec.tif bout_max_v_vec_v_vec.tif

oslc test_max_v_vec_u_vec.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_max_v_vec_u_vec.tif test_max_v_vec_u_vec
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_max_v_vec_u_vec.tif test_max_v_vec_u_vec
idiff sout_max_v_vec_u_vec.tif bout_max_v_vec_u_vec.tif

oslc test_max_v_dvec_v_dvec.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_max_v_dvec_v_dvec.tif test_max_v_dvec_v_dvec
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_max_v_dvec_v_dvec.tif test_max_v_dvec_v_dvec
idiff sout_max_v_dvec_v_dvec.tif bout_max_v_dvec_v_dvec.tif



