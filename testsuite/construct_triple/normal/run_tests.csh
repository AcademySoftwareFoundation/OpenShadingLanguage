rm *.tif *.oso

# normal float includes masking
oslc test_normal_u_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normal_u_float.tif test_normal_u_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normal_u_float.tif test_normal_u_float
idiff sout_normal_u_float.tif bout_normal_u_float.tif

oslc test_normal_v_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normal_v_float.tif test_normal_v_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normal_v_float.tif test_normal_v_float
idiff sout_normal_v_float.tif bout_normal_v_float.tif

# Derivs includes masking
oslc test_normal_v_dfloat.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_normal_v_dfloat.tif test_normal_v_dfloat
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normal_v_dfloat.tif test_normal_v_dfloat
idiff sout_normal_v_dfloat.tif bout_normal_v_dfloat.tif


# normal 3xfloat includes masking
oslc test_normal_3xu_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normal_3xu_float.tif test_normal_3xu_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normal_3xu_float.tif test_normal_3xu_float
idiff sout_normal_3xu_float.tif bout_normal_3xu_float.tif

oslc test_normal_3xv_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_normal_3xv_float.tif test_normal_3xv_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normal_3xv_float.tif test_normal_3xv_float
idiff sout_normal_3xv_float.tif bout_normal_3xv_float.tif

# Derivs includes masking
oslc test_normal_3xv_dfloat.osl
testshade --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout sout_normal_3xv_dfloat.tif test_normal_3xv_dfloat
testshade --vary_udxdy --vary_udxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_normal_3xv_dfloat.tif test_normal_3xv_dfloat
idiff sout_normal_3xv_dfloat.tif bout_normal_3xv_dfloat.tif



# normal normalspace 3x u float includes masking
oslc test_normal_u_space_3xu_float.osl
testshade -t 1 -g 64 64 -param normalspace common -od uint8 -o Cout sout_common_3xu_float.tif test_normal_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param normalspace common -od uint8 -o Cout bout_common_3xu_float.tif test_normal_u_space_3xu_float
idiff sout_common_3xu_float.tif bout_common_3xu_float.tif

testshade -t 1 -g 64 64 -param normalspace object -od uint8 -o Cout sout_object_3xu_float.tif test_normal_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param normalspace object -od uint8 -o Cout bout_object_3xu_float.tif test_normal_u_space_3xu_float
idiff sout_object_3xu_float.tif bout_object_3xu_float.tif

testshade -t 1 -g 64 64 -param normalspace shader -od uint8 -o Cout sout_shader_3xu_float.tif test_normal_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param normalspace shader -od uint8 -o Cout bout_shader_3xu_float.tif test_normal_u_space_3xu_float
idiff sout_shader_3xu_float.tif bout_shader_3xu_float.tif

testshade -t 1 -g 64 64 -param normalspace world -od uint8 -o Cout sout_world_3xu_float.tif test_normal_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param normalspace world -od uint8 -o Cout bout_world_3xu_float.tif test_normal_u_space_3xu_float
idiff sout_world_3xu_float.tif bout_world_3xu_float.tif

echo testshade doesn\'t define camara, screen, raster, or NDC spaces
testshade -t 1 -g 64 64 -param normalspace camera -od uint8 -o Cout sout_camera_3xu_float.tif test_normal_u_space_3xu_float
testshade -t 1 --batched -g 64 64 -param normalspace camera -od uint8 -o Cout bout_camera_3xu_float.tif test_normal_u_space_3xu_float
idiff sout_camera_3xu_float.tif bout_camera_3xu_float.tif



# normal normalspace 3x v float includes masking
oslc test_normal_u_space_3xv_float.osl
testshade -t 1 -g 64 64 -param normalspace common -od uint8 -o Cout sout_common_3xv_float.tif test_normal_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param normalspace common -od uint8 -o Cout bout_common_3xv_float.tif test_normal_u_space_3xv_float
idiff sout_common_3xv_float.tif bout_common_3xv_float.tif

testshade -t 1 -g 64 64 -param normalspace object -od uint8 -o Cout sout_object_3xv_float.tif test_normal_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param normalspace object -od uint8 -o Cout bout_object_3xv_float.tif test_normal_u_space_3xv_float
idiff sout_object_3xv_float.tif bout_object_3xv_float.tif

testshade -t 1 -g 64 64 -param normalspace shader -od uint8 -o Cout sout_shader_3xv_float.tif test_normal_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param normalspace shader -od uint8 -o Cout bout_shader_3xv_float.tif test_normal_u_space_3xv_float
idiff sout_shader_3xv_float.tif bout_shader_3xv_float.tif

testshade -t 1 -g 64 64 -param normalspace world -od uint8 -o Cout sout_world_3xv_float.tif test_normal_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param normalspace world -od uint8 -o Cout bout_world_3xv_float.tif test_normal_u_space_3xv_float
idiff sout_world_3xv_float.tif bout_world_3xv_float.tif

echo testshade doesn\'t define camara, screen, raster, or NDC spaces
testshade -t 1 -g 64 64 -param normalspace camera -od uint8 -o Cout sout_camera_3xv_float.tif test_normal_u_space_3xv_float
testshade -t 1 --batched -g 64 64 -param normalspace camera -od uint8 -o Cout bout_camera_3xv_float.tif test_normal_u_space_3xv_float
idiff sout_camera_3xv_float.tif bout_camera_3xv_float.tif




# normal normalspace 3x v dfloat includes masking
oslc test_normal_u_space_3xv_dfloat.osl
testshade -t 1 -g 64 64 -vary_udxdy --vary_udxdy -param normalspace common -od uint8 -o Cout sout_common_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -vary_udxdy --vary_udxdy -param normalspace common -od uint8 -o Cout bout_common_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
idiff sout_common_3xv_dfloat.tif bout_common_3xv_dfloat.tif

testshade -t 1 -g 64 64 -vary_udxdy --vary_udxdy -param normalspace object -od uint8 -o Cout sout_object_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -vary_udxdy --vary_udxdy -param normalspace object -od uint8 -o Cout bout_object_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
idiff sout_object_3xv_dfloat.tif bout_object_3xv_dfloat.tif

testshade -t 1 -g 64 64 -vary_udxdy --vary_udxdy -param normalspace shader -od uint8 -o Cout sout_shader_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -vary_udxdy --vary_udxdy -param normalspace shader -od uint8 -o Cout bout_shader_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
idiff sout_shader_3xv_dfloat.tif bout_shader_3xv_dfloat.tif

testshade -t 1 -g 64 64 -vary_udxdy --vary_udxdy -param normalspace world -od uint8 -o Cout sout_world_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -vary_udxdy --vary_udxdy -param normalspace world -od uint8 -o Cout bout_world_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
idiff sout_world_3xv_dfloat.tif bout_world_3xv_dfloat.tif

echo testshade doesn\'t define camara, screen, raster, or NDC spaces
testshade -t 1 -g 64 64 -vary_udxdy --vary_udxdy -param normalspace camera -od uint8 -o Cout sout_camera_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -vary_udxdy --vary_udxdy -param normalspace camera -od uint8 -o Cout bout_camera_3xv_dfloat.tif test_normal_u_space_3xv_dfloat
idiff sout_camera_3xv_dfloat.tif bout_camera_3xv_dfloat.tif


# normal varying normalspace 3x u float includes masking
oslc test_normal_v_space_3xu_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_space_3xu_float.tif test_normal_v_space_3xu_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_space_3xu_float.tif test_normal_v_space_3xu_float
idiff sout_v_space_3xu_float.tif bout_v_space_3xu_float.tif


# normal varying normalspace 3x v float includes masking
oslc test_normal_v_space_3xv_float.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_v_space_3xv_float.tif test_normal_v_space_3xv_float
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_v_space_3xv_float.tif test_normal_v_space_3xv_float
idiff sout_v_space_3xv_float.tif bout_v_space_3xv_float.tif

# normal varying normalspace 3x v dual float includes masking
# NOTE: current single normal impl just 0's the derivs out, tests are to make sure we don't miss a fix for that
#       So expect all deriv based outputs to be black images
oslc test_normal_v_space_3xv_dfloat.osl
testshade -t 1 -g 64 64 -vary_udxdy --vary_udxdy -od uint8 -o Cout sout_v_space_3xv_dfloat.tif test_normal_v_space_3xv_dfloat
testshade -t 1 --batched -g 64 64 -vary_udxdy --vary_udxdy -od uint8 -o Cout bout_v_space_3xv_dfloat.tif test_normal_v_space_3xv_dfloat
idiff sout_v_space_3xv_dfloat.tif bout_v_space_3xv_dfloat.tif




