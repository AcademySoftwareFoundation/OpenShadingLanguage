rm *.tif *.oso

# max(int, int) (includes masking)
oslc test_max_u_int_u_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_max_u_int_u_int.tif test_max_u_int_u_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_max_u_int_u_int.tif test_max_u_int_u_int
idiff sout_max_u_int_u_int.tif bout_max_u_int_u_int.tif

oslc test_max_u_int_v_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_max_u_int_v_int.tif test_max_u_int_v_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_max_u_int_v_int.tif test_max_u_int_v_int
idiff sout_max_u_int_v_int.tif bout_max_u_int_v_int.tif

oslc test_max_v_int_u_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_max_v_int_u_int.tif test_max_v_int_u_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_max_v_int_u_int.tif test_max_v_int_u_int
idiff sout_max_v_int_u_int.tif bout_max_v_int_u_int.tif

oslc test_max_v_int_v_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_max_v_int_v_int.tif test_max_v_int_v_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_max_v_int_v_int.tif test_max_v_int_v_int
idiff sout_max_v_int_v_int.tif bout_max_v_int_v_int.tif

