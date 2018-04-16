rm *.tif *.oso

oslc test_transform_u_tospace_u_normal.osl
oslc test_transform_u_tospace_v_normal.osl
oslc test_transform_u_tospace_v_dnormal.osl

./run_tospace_tests.csh common
./run_tospace_tests.csh object
./run_tospace_tests.csh shader
./run_tospace_tests.csh world
./run_tospace_tests.csh camera

# transform v tospace u normal includes masking
oslc test_transform_v_tospace_u_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_tospace_u_normal.tif test_transform_v_tospace_u_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_tospace_u_normal.tif test_transform_v_tospace_u_normal
idiff sout_transform_v_tospace_u_normal.tif bout_transform_v_tospace_u_normal.tif

# transform v tospace v normal includes masking
oslc test_transform_v_tospace_v_normal.osl
testshade -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_tospace_v_normal.tif test_transform_v_tospace_v_normal
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_tospace_v_normal.tif test_transform_v_tospace_v_normal
idiff sout_transform_v_tospace_v_normal.tif bout_transform_v_tospace_v_normal.tif

# transform v tospace v dual normal includes masking
oslc test_transform_v_tospace_v_dnormal.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout sout_transform_v_tospace_v_dnormal.tif test_transform_v_tospace_v_dnormal
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout bout_transform_v_tospace_v_dnormal.tif test_transform_v_tospace_v_dnormal
idiff sout_transform_v_tospace_v_dnormal.tif bout_transform_v_tospace_v_dnormal.tif
