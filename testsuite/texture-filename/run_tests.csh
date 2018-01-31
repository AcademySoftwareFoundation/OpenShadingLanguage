oslc test_filename_uniform.osl
testshade -g 64 64 --center -od uint8 -o Cout sout_filename_uniform.tif test_filename_uniform
testshade --batched -g 64 64 --center -od uint8 -o Cout bout_filename_uniform.tif test_filename_uniform
idiff sout_filename_uniform.tif bout_filename_uniform.tif


oslc test_filename_varying.osl
testshade -g 64 64 --center -od uint8 -o Cout sout_filename_varying.tif test_filename_varying
testshade --batched -g 64 64 --center -od uint8 -o Cout bout_filename_varying.tif test_filename_varying
idiff sout_filename_varying.tif bout_filename_varying.tif
