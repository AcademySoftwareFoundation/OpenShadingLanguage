oslc test_firstchannel_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_firstchannel_varying.tif test_firstchannel_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_firstchannel_varying.tif test_firstchannel_varying
idiff sout_firstchannel_varying.tif bout_firstchannel_varying.tif 
