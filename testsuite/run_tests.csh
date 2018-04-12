#!/bin/csh

./run_test_dir.csh normalize
./exec_test_dir.csh blendmath
./exec_test_dir.csh construct_triple
./run_test_dir.csh mxcompref
./run_test_dir.csh arraycopy

./run_test_dir.csh trig/pow
./run_test_dir.csh trig/atan2
./run_test_dir.csh trig/fmod
./run_test_dir.csh trig/sincos

./run_test_dir.csh vector/dot_tests
./run_test_dir.csh vector/distance_tests
./run_test_dir.csh vector/cross_tests

./run_test_dir.csh filterwidth

./run_test_dir.csh compassign

./run_test_dir.csh abs
./run_test_dir.csh neg

./run_test_dir.csh smoothstep
./run_test_dir.csh color

./run_test_dir.csh texture-filename
./run_test_dir.csh texture-width
./run_test_dir.csh texture-firstchannel
./run_test_dir.csh texture-missingcolor
./run_test_dir.csh texture-missingalpha
./run_test_dir.csh texture-subimage
./run_test_dir.csh texture-blur
./run_test_dir.csh texture-interp
./run_test_dir.csh texture-subimagename
./run_test_dir.csh texture-wrap

./run_test_dir.csh array 
./run_test_dir.csh array_assign
./run_test_dir.csh exit

./exec_test_dir.csh shaderglobals 
./exec_test_dir.csh breakcont 

./exec_test_dir.csh div 
./exec_test_dir.csh mul
./exec_test_dir.csh add
./exec_test_dir.csh sub
./exec_test_dir.csh max
./exec_test_dir.csh min

./run_test_dir.csh luminance
