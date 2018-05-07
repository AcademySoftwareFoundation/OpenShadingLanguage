oslc test_missingalpha_uniform.osl
testshade -g 64 64 --center -od uint8 -o Aout sout_alpha_missingalpha_uniform.tif test_missingalpha_uniform
testshade --batched -g 64 64 --center -od uint8 -o Aout bout_alpha_missingalpha_uniform.tif test_missingalpha_uniform
idiff sout_alpha_missingalpha_uniform.tif bout_alpha_missingalpha_uniform.tif


oslc test_missingalpha_varying.osl
testshade -g 64 64 --center -od uint8 -o Aout sout_alpha_missingalpha_varying.tif test_missingalpha_varying
testshade --batched -g 64 64 --center -od uint8 -o Aout bout_alpha_missingalpha_varying.tif test_missingalpha_varying
idiff sout_alpha_missingalpha_varying.tif bout_alpha_missingalpha_varying.tif
