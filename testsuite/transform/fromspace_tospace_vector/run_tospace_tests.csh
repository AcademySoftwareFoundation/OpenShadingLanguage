# transform v fromspace u tospace u vector includes masking
echo testshade -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_v_fromspace_$1_tospace_u_vector.tif" test_transform_v_fromspace_u_tospace_u_vector
testshade -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_v_fromspace_$1_tospace_u_vector.tif" test_transform_v_fromspace_u_tospace_u_vector

echo testshade -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_v_fromspace_$1_tospace_u_vector.tif" test_transform_v_fromspace_u_tospace_u_vector
testshade -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_v_fromspace_$1_tospace_u_vector.tif" test_transform_v_fromspace_u_tospace_u_vector
idiff "sout_transform_v_fromspace_$1_tospace_u_vector.tif" "bout_transform_v_fromspace_$1_tospace_u_vector.tif"


# transform v fromspace u tospace v vector includes masking
echo testshade -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_v_fromspace_$1_tospace_v_vector.tif" test_transform_v_fromspace_u_tospace_v_vector
testshade -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_v_fromspace_$1_tospace_v_vector.tif" test_transform_v_fromspace_u_tospace_v_vector

echo testshade -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_v_fromspace_$1_tospace_v_vector.tif" test_transform_v_fromspace_u_tospace_v_vector
testshade -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_v_fromspace_$1_tospace_v_vector.tif" test_transform_v_fromspace_u_tospace_v_vector
idiff "sout_transform_v_fromspace_$1_tospace_v_vector.tif" "bout_transform_v_fromspace_$1_tospace_v_vector.tif"


# transform v fromspace u tospace v dual vector includes masking
echo testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_v_fromspace_$1_tospace_v_dvector.tif" test_transform_v_fromspace_u_tospace_v_dvector
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_v_fromspace_$1_tospace_v_dvector.tif" test_transform_v_fromspace_u_tospace_v_dvector

echo testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_v_fromspace_$1_tospace_v_dvector.tif" test_transform_v_fromspace_u_tospace_v_dvector
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_v_fromspace_$1_tospace_v_dvector.tif" test_transform_v_fromspace_u_tospace_v_dvector
idiff "sout_transform_v_fromspace_$1_tospace_v_dvector.tif" "bout_transform_v_fromspace_$1_tospace_v_dvector.tif"




