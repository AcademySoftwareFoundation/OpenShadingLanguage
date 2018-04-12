rm *.tif *.oso

# transform v fromspace v tospace u point includes masking
oslc test_transform_v_fromspace_v_tospace_u_point.osl
echo testshade -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_u_point.tif" test_transform_v_fromspace_v_tospace_u_point
testshade -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_u_point.tif" test_transform_v_fromspace_v_tospace_u_point
echo testshade -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_u_point.tif" test_transform_v_fromspace_v_tospace_u_point
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_u_point.tif" test_transform_v_fromspace_v_tospace_u_point
idiff "sout_transform_v_fromspace_v_tospace_u_point.tif" "bout_transform_v_fromspace_v_tospace_u_point.tif"

# transform v fromspace v tospace v point includes masking
oslc test_transform_v_fromspace_v_tospace_v_point.osl
echo testshade -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_v_point.tif" test_transform_v_fromspace_v_tospace_v_point
testshade -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_v_point.tif" test_transform_v_fromspace_v_tospace_v_point
echo testshade -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_v_point.tif" test_transform_v_fromspace_v_tospace_v_point
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_v_point.tif" test_transform_v_fromspace_v_tospace_v_point
idiff "sout_transform_v_fromspace_v_tospace_v_point.tif" "bout_transform_v_fromspace_v_tospace_v_point.tif"

# transform v fromspace v tospace v dual point includes masking
oslc test_transform_v_fromspace_v_tospace_v_dpoint.osl
echo testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_v_dpoint.tif" test_transform_v_fromspace_v_tospace_v_dpoint
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_v_dpoint.tif" test_transform_v_fromspace_v_tospace_v_dpoint
echo testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_v_dpoint.tif" test_transform_v_fromspace_v_tospace_v_dpoint
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_v_dpoint.tif" test_transform_v_fromspace_v_tospace_v_dpoint
idiff "sout_transform_v_fromspace_v_tospace_v_dpoint.tif" "bout_transform_v_fromspace_v_tospace_v_dpoint.tif"

./run_fromspace_tests.csh common
./run_fromspace_tests.csh object
./run_fromspace_tests.csh shader
./run_fromspace_tests.csh world
./run_fromspace_tests.csh camera

./run_tospace_tests.csh common
./run_tospace_tests.csh object
./run_tospace_tests.csh shader
./run_tospace_tests.csh world
./run_tospace_tests.csh camera