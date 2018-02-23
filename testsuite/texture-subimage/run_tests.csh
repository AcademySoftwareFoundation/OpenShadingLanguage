oslc test_subimage_uniform.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_subimage_uniform.tif test_subimage_uniform
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_subimage_uniform.tif test_subimage_uniform
idiff sout_subimage_uniform.tif bout_subimage_uniform.tif 


oslc test_subimage_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_subimage_varying.tif test_subimage_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_subimage_varying.tif test_subimage_varying
idiff sout_subimage_varying.tif bout_subimage_varying.tif 
