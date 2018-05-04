./run_fromspace_tospace_tests.csh $1 common
./run_fromspace_tospace_tests.csh $1 object
./run_fromspace_tospace_tests.csh $1 shader
./run_fromspace_tospace_tests.csh $1 world
./run_fromspace_tospace_tests.csh $1 camera

# transform u fromspace v tospace u point includes masking
echo testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_u_point.tif" test_transform_u_fromspace_v_tospace_u_point
testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_u_point.tif" test_transform_u_fromspace_v_tospace_u_point
echo testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_u_point.tif" test_transform_u_fromspace_v_tospace_u_point
testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_u_point.tif" test_transform_u_fromspace_v_tospace_u_point
idiff "sout_transform_$1_fromspace_v_tospace_u_point.tif" "bout_transform_$1_fromspace_v_tospace_u_point.tif"

# transform u fromspace v tospace v point includes masking
echo testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_v_point.tif" test_transform_u_fromspace_v_tospace_v_point
testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_v_point.tif" test_transform_u_fromspace_v_tospace_v_point
echo testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_v_point.tif" test_transform_u_fromspace_v_tospace_v_point
testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_v_point.tif" test_transform_u_fromspace_v_tospace_v_point
idiff "sout_transform_$1_fromspace_v_tospace_v_point.tif" "bout_transform_$1_fromspace_v_tospace_v_point.tif"

# transform u fromspace v tospace v dual point includes masking
echo testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_v_dpoint.tif" test_transform_u_fromspace_v_tospace_v_dpoint
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_v_dpoint.tif" test_transform_u_fromspace_v_tospace_v_dpoint
echo testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_v_dpoint.tif" test_transform_u_fromspace_v_tospace_v_dpoint
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_v_dpoint.tif" test_transform_u_fromspace_v_tospace_v_dpoint
idiff "sout_transform_$1_fromspace_v_tospace_v_dpoint.tif" "bout_transform_$1_fromspace_v_tospace_v_dpoint.tif"
