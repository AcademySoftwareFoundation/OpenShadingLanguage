rm *.tif *.oso

# normalize vector includes masking
oslc test_normalize_u_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normalize_u_vector.tif test_normalize_u_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normalize_u_vector.tif test_normalize_u_vector
idiff sout_normalize_u_vector.tif bout_normalize_u_vector.tif


oslc test_normalize_v_vector.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normalize_v_vector.tif test_normalize_v_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normalize_v_vector.tif test_normalize_v_vector
idiff sout_normalize_v_vector.tif bout_normalize_v_vector.tif


oslc test_normalize_v_dvector.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_normalize_v_dvector.tif test_normalize_v_dvector
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normalize_v_dvector.tif test_normalize_v_dvector
idiff sout_normalize_v_dvector.tif bout_normalize_v_dvector.tif




# normalize normal includes masking
oslc test_normalize_u_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normalize_u_normal.tif test_normalize_u_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normalize_u_normal.tif test_normalize_u_normal
idiff sout_normalize_u_normal.tif bout_normalize_u_normal.tif


oslc test_normalize_v_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normalize_v_normal.tif test_normalize_v_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normalize_v_normal.tif test_normalize_v_normal
idiff sout_normalize_v_normal.tif bout_normalize_v_normal.tif


oslc test_normalize_v_dnormal.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_normalize_v_dnormal.tif test_normalize_v_dnormal
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normalize_v_dnormal.tif test_normalize_v_dnormal
idiff sout_normalize_v_dnormal.tif bout_normalize_v_dnormal.tif

