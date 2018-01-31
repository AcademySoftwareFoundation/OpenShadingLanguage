oslc test_width_uniform.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_width_uniform.tif test_width_uniform
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_width_uniform.tif test_width_uniform
idiff sout_width_uniform.tif bout_width_uniform.tif

oslc test_width_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_width_varying.tif test_width_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_width_varying.tif test_width_varying
idiff sout_width_varying.tif bout_width_varying.tif

