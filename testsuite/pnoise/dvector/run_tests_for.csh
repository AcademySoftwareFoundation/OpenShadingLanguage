echo Run dvector Tests for $1

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_1d_u_float_u_float.tif test_dvector_1d_u_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_1d_u_float_u_float.tif test_dvector_1d_u_float_u_float
idiff sout_$1_dvector_1d_u_float_u_float.tif bout_$1_dvector_1d_u_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_1d_v_float_u_float.tif test_dvector_1d_v_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_1d_v_float_u_float.tif test_dvector_1d_v_float_u_float
idiff -fail 0.04 sout_$1_dvector_1d_v_float_u_float.tif bout_$1_dvector_1d_v_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_1d_u_float_v_float.tif test_dvector_1d_u_float_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_1d_u_float_v_float.tif test_dvector_1d_u_float_v_float
idiff sout_$1_dvector_1d_u_float_v_float.tif bout_$1_dvector_1d_u_float_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_1d_v_float_v_float.tif test_dvector_1d_v_float_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_1d_v_float_v_float.tif test_dvector_1d_v_float_v_float
idiff sout_$1_dvector_1d_v_float_v_float.tif bout_$1_dvector_1d_v_float_v_float.tif






testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_u_float_u_float_u_float_u_float.tif test_dvector_2d_u_float_u_float_u_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_u_float_u_float_u_float_u_float.tif test_dvector_2d_u_float_u_float_u_float_u_float
idiff sout_$1_dvector_2d_u_float_u_float_u_float_u_float.tif bout_$1_dvector_2d_u_float_u_float_u_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_u_float_v_float_u_float_u_float.tif test_dvector_2d_u_float_v_float_u_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_u_float_v_float_u_float_u_float.tif test_dvector_2d_u_float_v_float_u_float_u_float
idiff sout_$1_dvector_2d_u_float_v_float_u_float_u_float.tif bout_$1_dvector_2d_u_float_v_float_u_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_v_float_u_float_u_float_u_float.tif test_dvector_2d_v_float_u_float_u_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_v_float_u_float_u_float_u_float.tif test_dvector_2d_v_float_u_float_u_float_u_float
idiff sout_$1_dvector_2d_v_float_u_float_u_float_u_float.tif bout_$1_dvector_2d_v_float_u_float_u_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif test_dvector_2d_v_float_v_float_u_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif test_dvector_2d_v_float_v_float_u_float_u_float
idiff sout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif bout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 4 4  -od uint8 -o Cout sout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif test_dvector_2d_v_float_v_float_u_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 4 4  -od uint8 -o Cout bout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif test_dvector_2d_v_float_v_float_u_float_u_float
idiff sout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif bout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif


testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_u_float_u_float_u_float_v_float.tif test_dvector_2d_u_float_u_float_u_float_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_u_float_u_float_u_float_v_float.tif test_dvector_2d_u_float_u_float_u_float_v_float
idiff sout_$1_dvector_2d_u_float_u_float_u_float_v_float.tif bout_$1_dvector_2d_u_float_u_float_u_float_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_u_float_v_float_u_float_u_float.tif test_dvector_2d_u_float_v_float_u_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_u_float_v_float_u_float_u_float.tif test_dvector_2d_u_float_v_float_u_float_u_float
idiff sout_$1_dvector_2d_u_float_v_float_u_float_u_float.tif bout_$1_dvector_2d_u_float_v_float_u_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_v_float_u_float_u_float_u_float.tif test_dvector_2d_v_float_u_float_u_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_v_float_u_float_u_float_u_float.tif test_dvector_2d_v_float_u_float_u_float_u_float
idiff sout_$1_dvector_2d_v_float_u_float_u_float_u_float.tif bout_$1_dvector_2d_v_float_u_float_u_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif test_dvector_2d_v_float_v_float_u_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif test_dvector_2d_v_float_v_float_u_float_u_float
idiff sout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif bout_$1_dvector_2d_v_float_v_float_u_float_u_float.tif



testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_u_float_u_float_v_float_u_float.tif test_dvector_2d_u_float_u_float_v_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_u_float_u_float_v_float_u_float.tif test_dvector_2d_u_float_u_float_v_float_u_float
idiff sout_$1_dvector_2d_u_float_u_float_v_float_u_float.tif bout_$1_dvector_2d_u_float_u_float_v_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_u_float_v_float_v_float_u_float.tif test_dvector_2d_u_float_v_float_v_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_u_float_v_float_v_float_u_float.tif test_dvector_2d_u_float_v_float_v_float_u_float
idiff -fail 0.04 sout_$1_dvector_2d_u_float_v_float_v_float_u_float.tif bout_$1_dvector_2d_u_float_v_float_v_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_v_float_u_float_v_float_u_float.tif test_dvector_2d_v_float_u_float_v_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_v_float_u_float_v_float_u_float.tif test_dvector_2d_v_float_u_float_v_float_u_float
idiff sout_$1_dvector_2d_v_float_u_float_v_float_u_float.tif bout_$1_dvector_2d_v_float_u_float_v_float_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_v_float_v_float_v_float_u_float.tif test_dvector_2d_v_float_v_float_v_float_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_v_float_v_float_v_float_u_float.tif test_dvector_2d_v_float_v_float_v_float_u_float
idiff sout_$1_dvector_2d_v_float_v_float_v_float_u_float.tif bout_$1_dvector_2d_v_float_v_float_v_float_u_float.tif



testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_u_float_u_float_v_float_v_float.tif test_dvector_2d_u_float_u_float_v_float_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_u_float_u_float_v_float_v_float.tif test_dvector_2d_u_float_u_float_v_float_v_float
idiff sout_$1_dvector_2d_u_float_u_float_v_float_v_float.tif bout_$1_dvector_2d_u_float_u_float_v_float_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_u_float_v_float_v_float_v_float.tif test_dvector_2d_u_float_v_float_v_float_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_u_float_v_float_v_float_v_float.tif test_dvector_2d_u_float_v_float_v_float_v_float
idiff sout_$1_dvector_2d_u_float_v_float_v_float_v_float.tif bout_$1_dvector_2d_u_float_v_float_v_float_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_v_float_u_float_v_float_v_float.tif test_dvector_2d_v_float_u_float_v_float_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_v_float_u_float_v_float_v_float.tif test_dvector_2d_v_float_u_float_v_float_v_float
idiff sout_$1_dvector_2d_v_float_u_float_v_float_v_float.tif bout_$1_dvector_2d_v_float_u_float_v_float_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_2d_v_float_v_float_v_float_v_float.tif test_dvector_2d_v_float_v_float_v_float_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_2d_v_float_v_float_v_float_v_float.tif test_dvector_2d_v_float_v_float_v_float_v_float
idiff -fail 0.04 sout_$1_dvector_2d_v_float_v_float_v_float_v_float.tif bout_$1_dvector_2d_v_float_v_float_v_float_v_float.tif







testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_3d_u_point_u_point.tif test_dvector_3d_u_point_u_point
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_3d_u_point_u_point.tif test_dvector_3d_u_point_u_point
idiff sout_$1_dvector_3d_u_point_u_point.tif bout_$1_dvector_3d_u_point_u_point.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_3d_v_point_u_point.tif test_dvector_3d_v_point_u_point
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_3d_v_point_u_point.tif test_dvector_3d_v_point_u_point
idiff sout_$1_dvector_3d_v_point_u_point.tif bout_$1_dvector_3d_v_point_u_point.tif



testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_3d_u_point_v_point.tif test_dvector_3d_u_point_v_point
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_3d_u_point_v_point.tif test_dvector_3d_u_point_v_point
idiff sout_$1_dvector_3d_u_point_v_point.tif bout_$1_dvector_3d_u_point_v_point.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_3d_v_point_v_point.tif test_dvector_3d_v_point_v_point
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_3d_v_point_v_point.tif test_dvector_3d_v_point_v_point
idiff --fail 0.004 sout_$1_dvector_3d_v_point_v_point.tif bout_$1_dvector_3d_v_point_v_point.tif






testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_u_point_u_float_u_point_u_float.tif test_dvector_4d_u_point_u_float_u_point_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_u_point_u_float_u_point_u_float.tif test_dvector_4d_u_point_u_float_u_point_u_float
idiff sout_$1_dvector_4d_u_point_u_float_u_point_u_float.tif bout_$1_dvector_4d_u_point_u_float_u_point_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_u_point_v_float_u_point_u_float.tif test_dvector_4d_u_point_v_float_u_point_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_u_point_v_float_u_point_u_float.tif test_dvector_4d_u_point_v_float_u_point_u_float
idiff sout_$1_dvector_4d_u_point_v_float_u_point_u_float.tif bout_$1_dvector_4d_u_point_v_float_u_point_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_v_point_u_float_u_point_u_float.tif test_dvector_4d_v_point_u_float_u_point_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_v_point_u_float_u_point_u_float.tif test_dvector_4d_v_point_u_float_u_point_u_float
idiff sout_$1_dvector_4d_v_point_u_float_u_point_u_float.tif bout_$1_dvector_4d_v_point_u_float_u_point_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_v_point_v_float_u_point_u_float.tif test_dvector_4d_v_point_v_float_u_point_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_v_point_v_float_u_point_u_float.tif test_dvector_4d_v_point_v_float_u_point_u_float
idiff sout_$1_dvector_4d_v_point_v_float_u_point_u_float.tif bout_$1_dvector_4d_v_point_v_float_u_point_u_float.tif



testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_u_point_u_float_u_point_v_float.tif test_dvector_4d_u_point_u_float_u_point_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_u_point_u_float_u_point_v_float.tif test_dvector_4d_u_point_u_float_u_point_v_float
idiff sout_$1_dvector_4d_u_point_u_float_u_point_v_float.tif bout_$1_dvector_4d_u_point_u_float_u_point_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_u_point_v_float_u_point_v_float.tif test_dvector_4d_u_point_v_float_u_point_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_u_point_v_float_u_point_v_float.tif test_dvector_4d_u_point_v_float_u_point_v_float
idiff sout_$1_dvector_4d_u_point_v_float_u_point_v_float.tif bout_$1_dvector_4d_u_point_v_float_u_point_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_v_point_u_float_u_point_v_float.tif test_dvector_4d_v_point_u_float_u_point_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_v_point_u_float_u_point_v_float.tif test_dvector_4d_v_point_u_float_u_point_v_float
idiff sout_$1_dvector_4d_v_point_u_float_u_point_v_float.tif bout_$1_dvector_4d_v_point_u_float_u_point_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_v_point_v_float_u_point_v_float.tif test_dvector_4d_v_point_v_float_u_point_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_v_point_v_float_u_point_v_float.tif test_dvector_4d_v_point_v_float_u_point_v_float
idiff sout_$1_dvector_4d_v_point_v_float_u_point_v_float.tif bout_$1_dvector_4d_v_point_v_float_u_point_v_float.tif



testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_u_point_u_float_v_point_u_float.tif test_dvector_4d_u_point_u_float_v_point_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_u_point_u_float_v_point_u_float.tif test_dvector_4d_u_point_u_float_v_point_u_float
idiff sout_$1_dvector_4d_u_point_u_float_v_point_u_float.tif bout_$1_dvector_4d_u_point_u_float_v_point_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_u_point_v_float_v_point_u_float.tif test_dvector_4d_u_point_v_float_v_point_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_u_point_v_float_v_point_u_float.tif test_dvector_4d_u_point_v_float_v_point_u_float
idiff sout_$1_dvector_4d_u_point_v_float_v_point_u_float.tif bout_$1_dvector_4d_u_point_v_float_v_point_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_v_point_u_float_v_point_u_float.tif test_dvector_4d_v_point_u_float_v_point_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_v_point_u_float_v_point_u_float.tif test_dvector_4d_v_point_u_float_v_point_u_float
idiff -fail 0.04 sout_$1_dvector_4d_v_point_u_float_v_point_u_float.tif bout_$1_dvector_4d_v_point_u_float_v_point_u_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_v_point_v_float_v_point_u_float.tif test_dvector_4d_v_point_v_float_v_point_u_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_v_point_v_float_v_point_u_float.tif test_dvector_4d_v_point_v_float_v_point_u_float
idiff -fail 0.04 sout_$1_dvector_4d_v_point_v_float_v_point_u_float.tif bout_$1_dvector_4d_v_point_v_float_v_point_u_float.tif



testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_u_point_u_float_v_point_v_float.tif test_dvector_4d_u_point_u_float_v_point_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_u_point_u_float_v_point_v_float.tif test_dvector_4d_u_point_u_float_v_point_v_float
idiff sout_$1_dvector_4d_u_point_u_float_v_point_v_float.tif bout_$1_dvector_4d_u_point_u_float_v_point_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_u_point_v_float_v_point_v_float.tif test_dvector_4d_u_point_v_float_v_point_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_u_point_v_float_v_point_v_float.tif test_dvector_4d_u_point_v_float_v_point_v_float
idiff sout_$1_dvector_4d_u_point_v_float_v_point_v_float.tif bout_$1_dvector_4d_u_point_v_float_v_point_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_v_point_u_float_v_point_v_float.tif test_dvector_4d_v_point_u_float_v_point_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_v_point_u_float_v_point_v_float.tif test_dvector_4d_v_point_u_float_v_point_v_float
idiff -fail 0.04 sout_$1_dvector_4d_v_point_u_float_v_point_v_float.tif bout_$1_dvector_4d_v_point_u_float_v_point_v_float.tif

testshade --param type $1 --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout sout_$1_dvector_4d_v_point_v_float_v_point_v_float.tif test_dvector_4d_v_point_v_float_v_point_v_float
testshade --param type $1 --batched --vary_udxdy --vary_vdxdy -t 1 -g 64 64  -od uint8 -o Cout bout_$1_dvector_4d_v_point_v_float_v_point_v_float.tif test_dvector_4d_v_point_v_float_v_point_v_float
idiff -fail 0.04 sout_$1_dvector_4d_v_point_v_float_v_point_v_float.tif bout_$1_dvector_4d_v_point_v_float_v_point_v_float.tif









