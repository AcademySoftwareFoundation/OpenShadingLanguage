# transform tospace u point includes masking
echo testshade -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_$1_tospace_u_point.tif" test_transform_u_tospace_u_point
testshade -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_$1_tospace_u_point.tif" test_transform_u_tospace_u_point

echo testshade -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_$1_tospace_u_point.tif" test_transform_u_tospace_u_point
testshade -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_$1_tospace_u_point.tif" test_transform_u_tospace_u_point
idiff "sout_transform_$1_tospace_u_point.tif" "bout_transform_$1_tospace_u_point.tif"


# transform u tospace v point includes masking
echo testshade -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_$1_tospace_v_point.tif" test_transform_u_tospace_v_point
testshade -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_$1_tospace_v_point.tif" test_transform_u_tospace_v_point

echo testshade -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_$1_tospace_v_point.tif" test_transform_u_tospace_v_point
testshade -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_$1_tospace_v_point.tif" test_transform_u_tospace_v_point
idiff "sout_transform_$1_tospace_v_point.tif" "bout_transform_$1_tospace_v_point.tif"


# transform u tospace v dual point includes masking
echo testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_$1_tospace_v_dpoint.tif" test_transform_u_tospace_v_dpoint
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param tospace $1 -od uint8 -o Cout "sout_transform_$1_tospace_v_dpoint.tif" test_transform_u_tospace_v_dpoint

echo testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_$1_tospace_v_dpoint.tif" test_transform_u_tospace_v_dpoint
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param tospace $1 -od uint8 -o Cout "bout_transform_$1_tospace_v_dpoint.tif" test_transform_u_tospace_v_dpoint
idiff "sout_transform_$1_tospace_v_dpoint.tif" "bout_transform_$1_tospace_v_dpoint.tif"




