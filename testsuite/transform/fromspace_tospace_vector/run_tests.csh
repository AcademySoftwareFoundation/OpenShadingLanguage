rm *.tif *.oso

# transform v fromspace v tospace u vector includes masking
oslc test_transform_v_fromspace_v_tospace_u_vector.osl
echo testshade -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_u_vector.tif" test_transform_v_fromspace_v_tospace_u_vector
testshade -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_u_vector.tif" test_transform_v_fromspace_v_tospace_u_vector
echo testshade -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_u_vector.tif" test_transform_v_fromspace_v_tospace_u_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_u_vector.tif" test_transform_v_fromspace_v_tospace_u_vector
idiff "sout_transform_v_fromspace_v_tospace_u_vector.tif" "bout_transform_v_fromspace_v_tospace_u_vector.tif"

# transform v fromspace v tospace v vector includes masking
oslc test_transform_v_fromspace_v_tospace_v_vector.osl
echo testshade -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_v_vector.tif" test_transform_v_fromspace_v_tospace_v_vector
testshade -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_v_vector.tif" test_transform_v_fromspace_v_tospace_v_vector
echo testshade -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_v_vector.tif" test_transform_v_fromspace_v_tospace_v_vector
testshade -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_v_vector.tif" test_transform_v_fromspace_v_tospace_v_vector
idiff "sout_transform_v_fromspace_v_tospace_v_vector.tif" "bout_transform_v_fromspace_v_tospace_v_vector.tif"

# transform v fromspace v tospace v dual vector includes masking
oslc test_transform_v_fromspace_v_tospace_v_dvector.osl
echo testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_v_dvector.tif" test_transform_v_fromspace_v_tospace_v_dvector
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -od uint8 -o Cout "sout_transform_v_fromspace_v_tospace_v_dvector.tif" test_transform_v_fromspace_v_tospace_v_dvector
echo testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_v_dvector.tif" test_transform_v_fromspace_v_tospace_v_dvector
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -od uint8 -o Cout "bout_transform_v_fromspace_v_tospace_v_dvector.tif" test_transform_v_fromspace_v_tospace_v_dvector
idiff "sout_transform_v_fromspace_v_tospace_v_dvector.tif" "bout_transform_v_fromspace_v_tospace_v_dvector.tif"

oslc test_transform_u_fromspace_u_tospace_u_vector.osl
oslc test_transform_u_fromspace_u_tospace_v_vector.osl
oslc test_transform_u_fromspace_u_tospace_v_dvector.osl

oslc test_transform_u_fromspace_v_tospace_u_vector.osl
oslc test_transform_u_fromspace_v_tospace_v_vector.osl
oslc test_transform_u_fromspace_v_tospace_v_dvector.osl

./run_fromspace_tests.csh common
./run_fromspace_tests.csh object
./run_fromspace_tests.csh shader
./run_fromspace_tests.csh world
./run_fromspace_tests.csh camera

oslc test_transform_v_fromspace_u_tospace_u_vector.osl
oslc test_transform_v_fromspace_u_tospace_v_vector.osl
oslc test_transform_v_fromspace_u_tospace_v_dvector.osl

./run_tospace_tests.csh common
./run_tospace_tests.csh object
./run_tospace_tests.csh shader
./run_tospace_tests.csh world
./run_tospace_tests.csh camera