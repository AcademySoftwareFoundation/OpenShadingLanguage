oslc test_subimagename_uniform.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_subimagename_uniform.tif test_subimagename_uniform
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_subimagename_uniform.tif test_subimagename_uniform
# bug in TextureSystem leaves unitialized values that get copied back over
# future version of OIIO will fix it, leave the diff comparison out until then
#idiff sout_subimagename_uniform.tif bout_subimagename_uniform.tif

oslc test_subimagename_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_subimagename_varying.tif test_subimagename_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_subimagename_varying.tif test_subimagename_varying
# bug in TextureSystem leaves unitialized values that get copied back over
# future version of OIIO will fix it, leave the diff comparison out until then
#idiff sout_subimagename_varying.tif bout_subimagename_varying.tif

