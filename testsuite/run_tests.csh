#!/bin/csh

./run_test_dir.csh trig/pow

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

