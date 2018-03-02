oslc test.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout.tif test
#idiff sout.tif ref/out.tif
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout.tif test
idiff sout.tif bout.tif


oslc test_aastep.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_aastep.tif test_aastep
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_aastep.tif test_aastep
idiff sout_aastep.tif bout_aastep.tif

oslc test_aastep_masked.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_aastep_masked.tif test_aastep_masked
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_aastep_masked.tif test_aastep_masked
idiff sout_aastep_masked.tif bout_aastep_masked.tif
