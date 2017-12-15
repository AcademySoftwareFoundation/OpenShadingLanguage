testshade --iters 200 --stats -t 1 -g 1024 1024 test_sin -od uint8 -o rgb sout_sin.tif > sout_sin.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_sin -od uint8 -o rgb bout_sin.tif > bout_sin.txt
idiff sout_sin.tif bout_sin.tif
grep "Run " sout_sin.txt
grep "Run " bout_sin.txt

testshade --iters 200 --stats -t 1 -g 1024 1024 test_cos -od uint8 -o rgb sout_cos.tif > sout_cos.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_cos -od uint8 -o rgb bout_cos.tif > bout_cos.txt
idiff sout_cos.tif bout_cos.tif
grep "Run " sout_cos.txt
grep "Run " bout_cos.txt

testshade --iters 200 --stats -t 1 -g 1024 1024 test_tan -od uint8 -o rgb sout_tan.tif > sout_tan.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_tan -od uint8 -o rgb bout_tan.tif > bout_tan.txt
idiff sout_tan.tif bout_tan.tif
grep "Run " sout_tan.txt
grep "Run " bout_tan.txt



testshade --iters 200 --stats -t 1 -g 1024 1024 test_asin -od uint8 -o rgb sout_asin.tif > sout_asin.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_asin -od uint8 -o rgb bout_asin.tif > bout_asin.txt
idiff sout_asin.tif bout_asin.tif
grep "Run " sout_asin.txt
grep "Run " bout_asin.txt

testshade --iters 200 --stats -t 1 -g 1024 1024 test_acos -od uint8 -o rgb sout_acos.tif > sout_acos.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_acos -od uint8 -o rgb bout_acos.tif > bout_acos.txt
idiff sout_acos.tif bout_acos.tif
grep "Run " sout_acos.txt
grep "Run " bout_acos.txt

testshade --iters 200 --stats -t 1 -g 1024 1024 test_atan -od uint8 -o rgb sout_atan.tif > sout_atan.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_atan -od uint8 -o rgb bout_atan.tif > bout_atan.txt
idiff sout_atan.tif bout_atan.tif
grep "Run " sout_atan.txt
grep "Run " bout_atan.txt

testshade --iters 200 --stats -t 1 -g 1024 1024 test_atan2 -od uint8 -o rgb sout_atan2.tif > sout_atan2.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_atan2 -od uint8 -o rgb bout_atan2.tif > bout_atan2.txt
idiff sout_atan2.tif bout_atan2.tif
grep "Run " sout_atan2.txt
grep "Run " bout_atan2.txt




testshade --iters 200 --stats -t 1 -g 1024 1024 test_sinh -od uint8 -o rgb sout_sinh.tif > sout_sinh.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_sinh -od uint8 -o rgb bout_sinh.tif > bout_sinh.txt
idiff sout_sinh.tif bout_sinh.tif
grep "Run " sout_sinh.txt
grep "Run " bout_sinh.txt

testshade --iters 200 --stats -t 1 -g 1024 1024 test_cosh -od uint8 -o rgb sout_cosh.tif > sout_cosh.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_cosh -od uint8 -o rgb bout_cosh.tif > bout_cosh.txt
idiff sout_cosh.tif bout_cosh.tif
grep "Run " sout_cosh.txt
grep "Run " bout_cosh.txt

testshade --iters 200 --stats -t 1 -g 1024 1024 test_tanh -od uint8 -o rgb sout_tanh.tif > sout_tanh.txt
testshade --iters 200 --batched --stats -t 1 -g 1024 1024 test_tanh -od uint8 -o rgb bout_tanh.tif > bout_tanh.txt
idiff sout_tanh.tif bout_tanh.tif
grep "Run " sout_tanh.txt
grep "Run " bout_tanh.txt

