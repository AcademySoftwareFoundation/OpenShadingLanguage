
# transform u fromspace u tospace u point includes masking
oslc test_transform_u_fromspace_u_tospace_u_point.osl
echo testshade -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_u_point.tif" test_transform_u_fromspace_u_tospace_u_point
testshade -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_u_point.tif" test_transform_u_fromspace_u_tospace_u_point
echo testshade -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_u_point.tif" test_transform_u_fromspace_u_tospace_u_point
testshade -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_u_point.tif" test_transform_u_fromspace_u_tospace_u_point
idiff "sout_transform_$1_fromspace_$2_tospace_u_point.tif" "bout_transform_$1_fromspace_$2_tospace_u_point.tif"


# transform u fromspace u tospace v point includes masking
oslc test_transform_u_fromspace_u_tospace_v_point.osl
testshade -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_v_point.tif" test_transform_u_fromspace_u_tospace_v_point
testshade -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_v_point.tif" test_transform_u_fromspace_u_tospace_v_point
idiff "sout_transform_$1_fromspace_$2_tospace_v_point.tif" "bout_transform_$1_fromspace_$2_tospace_v_point.tif"


# transform u fromspace u tospace v dual point includes masking
oslc test_transform_u_fromspace_u_tospace_v_dpoint.osl
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_v_dpoint.tif" test_transform_u_fromspace_u_tospace_v_dpoint
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_v_dpoint.tif" test_transform_u_fromspace_u_tospace_v_dpoint
idiff "sout_transform_$1_fromspace_$2_tospace_v_dpoint.tif" "bout_transform_$1_fromspace_$2_tospace_v_dpoint.tif"




