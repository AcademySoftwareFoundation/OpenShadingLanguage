oslc test_interpmode_uniform.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_interpmode_uniform.tif test_interpmode_uniform
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_interpmode_uniform.tif test_interpmode_uniform
idiff sout_interpmode_uniform.tif bout_interpmode_uniform.tif

oslc test_interpmode_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_interpmode_varying.tif test_interpmode_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_interpmode_varying.tif test_interpmode_varying
idiff sout_interpmode_varying.tif bout_interpmode_varying.tif

