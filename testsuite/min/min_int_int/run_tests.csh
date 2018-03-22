rm *.tif *.oso

# min(int, int) (includes masking)
oslc test_min_u_int_u_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_u_int_u_int.tif test_min_u_int_u_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_u_int_u_int.tif test_min_u_int_u_int
idiff sout_min_u_int_u_int.tif bout_min_u_int_u_int.tif

oslc test_min_u_int_v_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_u_int_v_int.tif test_min_u_int_v_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_u_int_v_int.tif test_min_u_int_v_int
idiff sout_min_u_int_v_int.tif bout_min_u_int_v_int.tif

oslc test_min_v_int_u_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_v_int_u_int.tif test_min_v_int_u_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_v_int_u_int.tif test_min_v_int_u_int
idiff sout_min_v_int_u_int.tif bout_min_v_int_u_int.tif

oslc test_min_v_int_v_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_min_v_int_v_int.tif test_min_v_int_v_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_min_v_int_v_int.tif test_min_v_int_v_int
idiff sout_min_v_int_v_int.tif bout_min_v_int_v_int.tif

