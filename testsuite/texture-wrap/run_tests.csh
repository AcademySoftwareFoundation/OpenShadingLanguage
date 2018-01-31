oslc test_wrap_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_wrap_varying.tif test_wrap_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_wrap_varying.tif test_wrap_varying
idiff sout_wrap_varying.tif bout_wrap_varying.tif 

oslc test_swrap_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_swrap_varying.tif test_swrap_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_swrap_varying.tif test_swrap_varying
idiff sout_swrap_varying.tif bout_swrap_varying.tif

oslc test_twrap_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_twrap_varying.tif test_twrap_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_twrap_varying.tif test_twrap_varying
idiff sout_twrap_varying.tif bout_twrap_varying.tif 

oslc test_stwrap_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_stwrap_varying.tif test_stwrap_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_stwrap_varying.tif test_stwrap_varying
idiff sout_stwrap_varying.tif bout_stwrap_varying.tif 
