oslc test_missingcolor_uniform.osl
testshade -g 64 64 --center -od uint8 -o Cout sout_missingcolor_uniform.tif test_missingcolor_uniform
testshade --batched -g 64 64 --center -od uint8 -o Cout bout_missingcolor_uniform.tif test_missingcolor_uniform
idiff sout_missingcolor_uniform.tif bout_missingcolor_uniform.tif


oslc test_missingcolor_varying.osl
testshade -g 64 64 --center -od uint8 -o Cout sout_missingcolor_varying.tif test_missingcolor_varying
testshade --batched -g 64 64 --center -od uint8 -o Cout bout_missingcolor_varying.tif test_missingcolor_varying
idiff sout_missingcolor_varying.tif bout_missingcolor_varying.tif
