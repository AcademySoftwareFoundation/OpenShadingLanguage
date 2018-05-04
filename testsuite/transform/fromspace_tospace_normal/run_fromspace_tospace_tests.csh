
# transform u fromspace u tospace u normal includes masking
echo testshade -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_u_normal.tif" test_transform_u_fromspace_u_tospace_u_normal
testshade -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_u_normal.tif" test_transform_u_fromspace_u_tospace_u_normal

echo testshade -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_u_normal.tif" test_transform_u_fromspace_u_tospace_u_normal
testshade -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_u_normal.tif" test_transform_u_fromspace_u_tospace_u_normal
idiff "sout_transform_$1_fromspace_$2_tospace_u_normal.tif" "bout_transform_$1_fromspace_$2_tospace_u_normal.tif"


# transform u fromspace u tospace v normal includes masking
echo testshade -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_v_normal.tif" test_transform_u_fromspace_u_tospace_v_normal
testshade -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_v_normal.tif" test_transform_u_fromspace_u_tospace_v_normal

echo testshade -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_v_normal.tif" test_transform_u_fromspace_u_tospace_v_normal
testshade -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_v_normal.tif" test_transform_u_fromspace_u_tospace_v_normal
idiff "sout_transform_$1_fromspace_$2_tospace_v_normal.tif" "bout_transform_$1_fromspace_$2_tospace_v_normal.tif"


# transform u fromspace u tospace v dual normal includes masking
echo testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_v_dnormal.tif" test_transform_u_fromspace_u_tospace_v_dnormal
testshade --vary_udxdy --vary_vdxdy -t 1 -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "sout_transform_$1_fromspace_$2_tospace_v_dnormal.tif" test_transform_u_fromspace_u_tospace_v_dnormal

echo testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_v_dnormal.tif" test_transform_u_fromspace_u_tospace_v_dnormal
testshade --vary_udxdy --vary_vdxdy -t 1 --batched -g 64 64 -param fromspace $1 -param tospace $2 -od uint8 -o Cout "bout_transform_$1_fromspace_$2_tospace_v_dnormal.tif" test_transform_u_fromspace_u_tospace_v_dnormal
idiff "sout_transform_$1_fromspace_$2_tospace_v_dnormal.tif" "bout_transform_$1_fromspace_$2_tospace_v_dnormal.tif"




