rm *.tif *.oso

# smoothstep u float u float u float includes masking
oslc test_smoothstep_u_float_u_float_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_smoothstep_u_float_u_float_u_float.tif test_smoothstep_u_float_u_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_smoothstep_u_float_u_float_u_float.tif test_smoothstep_u_float_u_float_u_float
idiff sout_smoothstep_u_float_u_float_u_float.tif bout_smoothstep_u_float_u_float_u_float.tif


# smoothstep u float u float v float includes masking
oslc test_smoothstep_u_float_u_float_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_smoothstep_u_float_u_float_v_float.tif test_smoothstep_u_float_u_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_smoothstep_u_float_u_float_v_float.tif test_smoothstep_u_float_u_float_v_float
idiff sout_smoothstep_u_float_u_float_v_float.tif bout_smoothstep_u_float_u_float_v_float.tif


# smoothstep u float v float u float includes masking
oslc test_smoothstep_u_float_v_float_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_smoothstep_u_float_v_float_u_float.tif test_smoothstep_u_float_v_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_smoothstep_u_float_v_float_u_float.tif test_smoothstep_u_float_v_float_u_float
idiff sout_smoothstep_u_float_v_float_u_float.tif bout_smoothstep_u_float_v_float_u_float.tif


# smoothstep v float u float u float includes masking
oslc test_smoothstep_v_float_u_float_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_smoothstep_v_float_u_float_u_float.tif test_smoothstep_v_float_u_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_smoothstep_v_float_u_float_u_float.tif test_smoothstep_v_float_u_float_u_float
idiff sout_smoothstep_v_float_u_float_u_float.tif bout_smoothstep_v_float_u_float_u_float.tif



# smoothstep v float v float u float includes masking
oslc test_smoothstep_v_float_v_float_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_smoothstep_v_float_v_float_u_float.tif test_smoothstep_v_float_v_float_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_smoothstep_v_float_v_float_u_float.tif test_smoothstep_v_float_v_float_u_float
idiff sout_smoothstep_v_float_v_float_u_float.tif bout_smoothstep_v_float_v_float_u_float.tif

# smoothstep v float u float v float includes masking
oslc test_smoothstep_v_float_u_float_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_smoothstep_v_float_u_float_v_float.tif test_smoothstep_v_float_u_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_smoothstep_v_float_u_float_v_float.tif test_smoothstep_v_float_u_float_v_float
idiff sout_smoothstep_v_float_u_float_v_float.tif bout_smoothstep_v_float_u_float_v_float.tif

# smoothstep u float v float v float includes masking
oslc test_smoothstep_u_float_v_float_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_smoothstep_u_float_v_float_v_float.tif test_smoothstep_u_float_v_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_smoothstep_u_float_v_float_v_float.tif test_smoothstep_u_float_v_float_v_float
idiff sout_smoothstep_u_float_v_float_v_float.tif bout_smoothstep_u_float_v_float_v_float.tif

# smoothstep v float v float v float includes masking
oslc test_smoothstep_v_float_v_float_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_smoothstep_v_float_v_float_v_float.tif test_smoothstep_v_float_v_float_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_smoothstep_v_float_v_float_v_float.tif test_smoothstep_v_float_v_float_v_float
idiff sout_smoothstep_v_float_v_float_v_float.tif bout_smoothstep_v_float_v_float_v_float.tif
