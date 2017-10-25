rm sout.tif bout.tif ; oslc test_vary3.osl ; testshade -t 1 -g 8 2 test_vary3 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary3 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.3.log
rm sout.tif bout.tif ; oslc test_vary3b.osl ; testshade -t 1 -g 8 2 test_vary3b -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary3b -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.3b.log
rm sout.tif bout.tif ; oslc test_vary3c.osl ; testshade -t 1 -g 8 2 test_vary3c -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary3c -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.3c.log
rm sout.tif bout.tif ; oslc test_vary3d.osl ; testshade -t 1 -g 8 2 test_vary3d -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary3d -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.3d.log
rm sout.tif bout.tif ; oslc test_vary4.osl ; testshade -t 1 -g 8 2 test_vary4 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary4 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.4.log
rm sout.tif bout.tif ; oslc test_vary5.osl ; testshade -t 1 -g 8 2 test_vary5 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary5 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.5.log
rm sout.tif bout.tif ; oslc test_vary6.osl ; testshade -t 1 -g 8 2 test_vary6 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary6 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.6.log
rm sout.tif bout.tif ; oslc test_vary7.osl ; testshade -t 1 -g 8 2 test_vary7 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary7 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.7.log
rm sout.tif bout.tif ; oslc test_vary7b.osl ; testshade -t 1 -g 8 2 test_vary7b -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary7b -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.7b.log
rm sout.tif bout.tif ; oslc test_vary7c.osl ; testshade -t 1 -g 8 2 test_vary7c -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary7c -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.7c.log
rm sout.tif bout.tif ; oslc test_vary7d.osl ; testshade -t 1 -g 8 2 test_vary7d -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary7d -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.7d.log
rm sout.tif bout.tif ; oslc test_vary8.osl ; testshade -t 1 -g 8 2 test_vary8 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary8 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.8.log
rm sout.tif bout.tif ; oslc test_vary8b.osl ; testshade -t 1 -g 8 2 test_vary8b -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary8b -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.8b.log
rm sout.tif bout.tif ; oslc test_vary3e.osl ; testshade -t 1 -g 8 2 test_vary3e -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary3e -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.3e.log
rm sout.tif bout.tif ; oslc test_vary3f.osl ; testshade -t 1 -g 8 2 test_vary3f -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary3f -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.3f.log
rm sout.tif bout.tif ; oslc layer_a.osl; oslc layer_b.osl; testshade -t 1 -g 8 2 -layer alayer a --layer blayer b --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb sout.tif; testshade --batched -t 1 -g 8 2 -layer alayer a --layer blayer b --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb bout.tif; idiff sout.tif bout.tif > idiff.layer_ab.log
rm sout.tif bout.tif ; oslc layer_a2.osl; oslc layer_b2.osl; testshade -t 1 -g 8 2 -layer alayer a2 --layer blayer b2 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb sout.tif; testshade --batched -t 1 -g 8 2 -layer alayer a2 --layer blayer b2 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb bout.tif; idiff sout.tif bout.tif > idiff.layer_a2b2.log
rm sout.tif bout.tif ; oslc layer_a3.osl; oslc layer_b3.osl; testshade -t 1 -g 8 2 -layer alayer a3 --layer blayer b3 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb sout.tif; testshade --batched -t 1 -g 8 2 -layer alayer a3 --layer blayer b3 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb bout.tif; idiff sout.tif bout.tif > idiff.layer_a3b3.log

rm sout.tif bout.tif ; oslc layer_a4.osl; oslc layer_b4.osl; testshade -t 1 -g 8 2 -layer alayer a4 --layer blayer b4 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb sout.tif; testshade --batched -t 1 -g 8 2 -layer alayer a4 --layer blayer b4 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb bout.tif; idiff sout.tif bout.tif > idiff.layer_a4b4.log

rm sout.tif bout.tif ; oslc test_vary9.osl ; testshade -t 1 -g 8 2 test_vary9 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary9 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.9.log

rm sout.tif bout.tif ; oslc test_vary10.osl ; testshade -t 1 -g 8 2 test_vary10 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary10 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.10.log

rm sout.tif bout.tif ; oslc test_vary11.osl ; testshade -t 1 -g 8 2 test_vary11 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary11 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.11.log

rm sout.tif bout.tif ; oslc test_vary12.osl ; testshade -t 1 -g 8 2 test_vary12 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary12 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.12.log

rm sout.tif bout.tif ; oslc test_vary13.osl ; testshade -t 1 -g 8 2 test_vary13 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary13 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.13.log

rm sout.tif bout.tif ; oslc test_vary14.osl ; testshade -t 1 -g 8 2 test_vary14 -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary14 -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.14.log

rm sout.tif bout.tif ; oslc test_vary14b.osl ; testshade -t 1 -g 8 2 test_vary14b -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary14b -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.14b.log

rm sout.tif bout.tif ; oslc test_vary14c.osl ; testshade -t 1 -g 8 2 test_vary14c -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary14c -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.14c.log

rm sout.tif bout.tif ; oslc test_vary14d.osl ; testshade -t 1 -g 8 2 test_vary14d -od uint8 -o c sout.tif ; testshade --batched -t 1 -g 8 2 test_vary14d -od uint8 -o c bout.tif ; idiff sout.tif bout.tif > idiff.14d.log


echo idiff.3.log
cat idiff.3.log
echo idiff.3b.log
cat idiff.3b.log
echo idiff.3c.log
cat idiff.3c.log
echo idiff.3d.log
cat idiff.3d.log
echo idiff.4.log
cat idiff.4.log
echo idiff.5.log
cat idiff.5.log
echo idiff.6.log
cat idiff.6.log
echo idiff.7.log
cat idiff.7.log
echo idiff.7b.log
cat idiff.7b.log
echo idiff.7c.log
cat idiff.7c.log
echo idiff.7d.log
cat idiff.7d.log
echo idiff.8.log
cat idiff.8.log
echo idiff.8b.log
cat idiff.8b.log
echo idiff.3e.log
cat idiff.3e.log
echo idiff.3f.log
cat idiff.3f.log
echo idiff.layer_ab.log
cat idiff.layer_ab.log
echo idiff.layer_a2b2.log
cat idiff.layer_a2b2.log
echo idiff.layer_a3b3.log
cat idiff.layer_a3b3.log
echo idiff.layer_a4b4.log
cat idiff.layer_a4b4.log
echo idiff.9.log
cat idiff.9.log
echo idiff.10.log
cat idiff.10.log
echo idiff.11.log
cat idiff.11.log
echo idiff.12.log
cat idiff.12.log
echo idiff.13.log
cat idiff.13.log
echo idiff.14.log
cat idiff.14.log
echo idiff.14b.log
cat idiff.14b.log
echo idiff.14c.log
cat idiff.14c.log
echo idiff.14d.log
cat idiff.14d.log

