rm *.tif *.oso

# int - int (includes masking)
oslc test_u_int_sub_u_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_int_sub_u_int.tif test_u_int_sub_u_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_int_sub_u_int.tif test_u_int_sub_u_int
idiff sout_u_int_sub_u_int.tif bout_u_int_sub_u_int.tif

oslc test_u_int_sub_v_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_int_sub_v_int.tif test_u_int_sub_v_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_int_sub_v_int.tif test_u_int_sub_v_int
idiff sout_u_int_sub_v_int.tif bout_u_int_sub_v_int.tif

oslc test_v_int_sub_u_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_int_sub_u_int.tif test_v_int_sub_u_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_int_sub_u_int.tif test_v_int_sub_u_int
idiff sout_v_int_sub_u_int.tif bout_v_int_sub_u_int.tif

oslc test_v_int_sub_v_int.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_int_sub_v_int.tif test_v_int_sub_v_int
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_int_sub_v_int.tif test_v_int_sub_v_int
idiff sout_v_int_sub_v_int.tif bout_v_int_sub_v_int.tif

