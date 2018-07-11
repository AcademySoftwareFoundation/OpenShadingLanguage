rm *.tif *.oso

# color color int includes masking
oslc test_u_color_u_color_u_color.osl
oslc test_v_color_u_color_u_color.osl
oslc test_u_color_v_color_u_color.osl
oslc test_v_color_v_color_u_color.osl

oslc test_u_color_u_color_v_color.osl
oslc test_v_color_u_color_v_color.osl
oslc test_u_color_v_color_v_color.osl
oslc test_v_color_v_color_v_color.osl

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_color_u_color_u_color.tif test_u_color_u_color_u_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_color_u_color_u_color.tif test_u_color_u_color_u_color
idiff sout_u_color_u_color_u_color.tif bout_u_color_u_color_u_color.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_color_u_color_u_color.tif test_v_color_u_color_u_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_color_u_color_u_color.tif test_v_color_u_color_u_color
idiff sout_v_color_u_color_u_color.tif bout_v_color_u_color_u_color.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_color_v_color_u_color.tif test_u_color_v_color_u_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_color_v_color_u_color.tif test_u_color_v_color_u_color
idiff sout_u_color_v_color_u_color.tif bout_u_color_v_color_u_color.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_color_v_color_u_color.tif test_v_color_v_color_u_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_color_v_color_u_color.tif test_v_color_v_color_u_color
idiff sout_v_color_v_color_u_color.tif bout_v_color_v_color_u_color.tif


testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_color_u_color_v_color.tif test_u_color_u_color_v_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_color_u_color_v_color.tif test_u_color_u_color_v_color
idiff sout_u_color_u_color_v_color.tif bout_u_color_u_color_v_color.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_color_u_color_v_color.tif test_v_color_u_color_v_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_color_u_color_v_color.tif test_v_color_u_color_v_color
idiff sout_v_color_u_color_v_color.tif bout_v_color_u_color_v_color.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_u_color_v_color_v_color.tif test_u_color_v_color_v_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_u_color_v_color_v_color.tif test_u_color_v_color_v_color
idiff sout_u_color_v_color_v_color.tif bout_u_color_v_color_v_color.tif

testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_color_v_color_v_color.tif test_v_color_v_color_v_color
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_color_v_color_v_color.tif test_v_color_v_color_v_color
idiff sout_v_color_v_color_v_color.tif bout_v_color_v_color_v_color.tif