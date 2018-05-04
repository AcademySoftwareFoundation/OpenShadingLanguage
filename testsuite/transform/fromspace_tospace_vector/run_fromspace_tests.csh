./run_fromspace_tospace_tests.csh $1 common
./run_fromspace_tospace_tests.csh $1 object
./run_fromspace_tospace_tests.csh $1 shader
./run_fromspace_tospace_tests.csh $1 world
./run_fromspace_tospace_tests.csh $1 camera

# transform u fromspace v tospace u vector includes masking
echo testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_u_vector.tif" test_transform_u_fromspace_v_tospace_u_vector
testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_u_vector.tif" test_transform_u_fromspace_v_tospace_u_vector
echo testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_u_vector.tif" test_transform_u_fromspace_v_tospace_u_vector
testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_u_vector.tif" test_transform_u_fromspace_v_tospace_u_vector
idiff "sout_transform_$1_fromspace_v_tospace_u_vector.tif" "bout_transform_$1_fromspace_v_tospace_u_vector.tif"

# transform u fromspace v tospace v vector includes masking
echo testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_v_vector.tif" test_transform_u_fromspace_v_tospace_v_vector
testshade -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_v_vector.tif" test_transform_u_fromspace_v_tospace_v_vector
echo testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_v_vector.tif" test_transform_u_fromspace_v_tospace_v_vector
testshade -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_v_vector.tif" test_transform_u_fromspace_v_tospace_v_vector
idiff "sout_transform_$1_fromspace_v_tospace_v_vector.tif" "bout_transform_$1_fromspace_v_tospace_v_vector.tif"

# transform u fromspace v tospace v dual vector includes masking
echo testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_v_dvector.tif" test_transform_u_fromspace_v_tospace_v_dvector
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param fromspace $1 -od uint8 -o Cout "sout_transform_$1_fromspace_v_tospace_v_dvector.tif" test_transform_u_fromspace_v_tospace_v_dvector
echo testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_v_dvector.tif" test_transform_u_fromspace_v_tospace_v_dvector
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param fromspace $1 -od uint8 -o Cout "bout_transform_$1_fromspace_v_tospace_v_dvector.tif" test_transform_u_fromspace_v_tospace_v_dvector
idiff "sout_transform_$1_fromspace_v_tospace_v_dvector.tif" "bout_transform_$1_fromspace_v_tospace_v_dvector.tif"
