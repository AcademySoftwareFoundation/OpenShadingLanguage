oslc test_blur_uniform.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_blur_uniform.tif test_blur_uniform
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_blur_uniform.tif test_blur_uniform
idiff sout_blur_uniform.tif bout_blur_uniform.tif 

oslc test_blur_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_blur_varying.tif test_blur_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_blur_varying.tif test_blur_varying
idiff sout_blur_varying.tif bout_blur_varying.tif 

oslc test_sblur_uniform.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_sblur_uniform.tif test_sblur_uniform
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_sblur_uniform.tif test_sblur_uniform
idiff sout_sblur_uniform.tif bout_sblur_uniform.tif 

oslc test_tblur_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_tblur_varying.tif test_tblur_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_tblur_varying.tif test_tblur_varying
idiff sout_tblur_varying.tif bout_tblur_varying.tif 

oslc test_blurs_uniform.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_blurs_uniform.tif test_blurs_uniform
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_blurs_uniform.tif test_blurs_uniform
idiff sout_blurs_uniform.tif bout_blurs_uniform.tif 

oslc test_blurs_varying.osl
testshade -g 256 256 --center -od uint8 -o Cout sout_blurs_varying.tif test_blurs_varying
testshade --batched -g 256 256 --center -od uint8 -o Cout bout_blurs_varying.tif test_blurs_varying
idiff sout_blurs_varying.tif bout_blurs_varying.tif 
