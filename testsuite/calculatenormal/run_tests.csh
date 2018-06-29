rm *.tif *.oso

# calculatenormal u point includes masking
oslc test_calculatenormal_u_point.osl
testshade -t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout sout_calculatenormal_u_point.tif test_calculatenormal_u_point
testshade -t 1 --batched -g 64 64 --vary_pdxdy -od uint8 -o Cout bout_calculatenormal_u_point.tif test_calculatenormal_u_point
idiff sout_calculatenormal_u_point.tif bout_calculatenormal_u_point.tif

# calculatenormal v point includes masking
oslc test_calculatenormal_v_point.osl
testshade -t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout sout_calculatenormal_v_point.tif test_calculatenormal_v_point
testshade -t 1 --batched -g 64 64 --vary_pdxdy -od uint8 -o Cout bout_calculatenormal_v_point.tif test_calculatenormal_v_point
idiff sout_calculatenormal_v_point.tif bout_calculatenormal_v_point.tif

# calculatenormal v dpoint includes masking
oslc test_calculatenormal_dpoint.osl
testshade -t 1 -g 64 64 --vary_pdxdy -od uint8 -o Cout sout_calculatenormal_dpoint.tif test_calculatenormal_dpoint
testshade -t 1 --batched -g 64 64 --vary_pdxdy -od uint8 -o Cout bout_calculatenormal_dpoint.tif test_calculatenormal_dpoint
idiff sout_calculatenormal_dpoint.tif bout_calculatenormal_dpoint.tif

