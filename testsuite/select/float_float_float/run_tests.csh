rm *.tif *.oso

# float float int includes masking
oslc test_u_float_u_float_u_float.osl
oslc test_v_float_u_float_u_float.osl
oslc test_u_float_v_float_u_float.osl
oslc test_v_float_v_float_u_float.osl

oslc test_u_float_u_float_v_float.osl
oslc test_v_float_u_float_v_float.osl
oslc test_u_float_v_float_v_float.osl
oslc test_v_float_v_float_v_float.osl

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_float_u_float_u_float.tif test_u_float_u_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_float_u_float_u_float.tif test_u_float_u_float_u_float
idiff sout_u_float_u_float_u_float.tif bout_u_float_u_float_u_float.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_float_u_float_u_float.tif test_v_float_u_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_float_u_float_u_float.tif test_v_float_u_float_u_float
idiff sout_v_float_u_float_u_float.tif bout_v_float_u_float_u_float.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_float_v_float_u_float.tif test_u_float_v_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_float_v_float_u_float.tif test_u_float_v_float_u_float
idiff sout_u_float_v_float_u_float.tif bout_u_float_v_float_u_float.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_float_v_float_u_float.tif test_v_float_v_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_float_v_float_u_float.tif test_v_float_v_float_u_float
idiff sout_v_float_v_float_u_float.tif bout_v_float_v_float_u_float.tif


testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_float_u_float_v_float.tif test_u_float_u_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_float_u_float_v_float.tif test_u_float_u_float_v_float
idiff sout_u_float_u_float_v_float.tif bout_u_float_u_float_v_float.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_float_u_float_v_float.tif test_v_float_u_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_float_u_float_v_float.tif test_v_float_u_float_v_float
idiff sout_v_float_u_float_v_float.tif bout_v_float_u_float_v_float.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_float_v_float_v_float.tif test_u_float_v_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_float_v_float_v_float.tif test_u_float_v_float_v_float
idiff sout_u_float_v_float_v_float.tif bout_u_float_v_float_v_float.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_float_v_float_v_float.tif test_v_float_v_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_float_v_float_v_float.tif test_v_float_v_float_v_float
idiff sout_v_float_v_float_v_float.tif bout_v_float_v_float_v_float.tif