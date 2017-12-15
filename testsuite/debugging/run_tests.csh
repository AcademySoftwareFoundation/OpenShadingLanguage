rm sout.tif bout.tif ; oslc layer_a.osl; oslc layer_b.osl; testshade -t 1 -g 8 2 -layer alayer a --layer blayer b --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb sout.tif; testshade --batched -t 1 -g 8 2 -layer alayer a --layer blayer b --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb bout.tif; idiff sout.tif bout.tif > idiff.layer_ab.log
rm sout.tif bout.tif ; oslc layer_a2.osl; oslc layer_b2.osl; testshade -t 1 -g 8 2 -layer alayer a2 --layer blayer b2 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb sout.tif; testshade --batched -t 1 -g 8 2 -layer alayer a2 --layer blayer b2 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb bout.tif; idiff sout.tif bout.tif > idiff.layer_a2b2.log
rm sout.tif bout.tif ; oslc layer_a3.osl; oslc layer_b3.osl; testshade -t 1 -g 8 2 -layer alayer a3 --layer blayer b3 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb sout.tif; testshade --batched -t 1 -g 8 2 -layer alayer a3 --layer blayer b3 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb bout.tif; idiff sout.tif bout.tif > idiff.layer_a3b3.log

rm sout.tif bout.tif ; oslc layer_a4.osl; oslc layer_b4.osl; testshade -t 1 -g 8 2 -layer alayer a4 --layer blayer b4 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb sout.tif; testshade --batched -t 1 -g 8 2 -layer alayer a4 --layer blayer b4 --connect alayer f_out blayer f_in --connect alayer c_out blayer c_in -od uint8 -o c_outb bout.tif; idiff sout.tif bout.tif > idiff.layer_a4b4.log


echo idiff.layer_ab.log
cat idiff.layer_ab.log
echo idiff.layer_a2b2.log
cat idiff.layer_a2b2.log
echo idiff.layer_a3b3.log
cat idiff.layer_a3b3.log
echo idiff.layer_a4b4.log
cat idiff.layer_a4b4.log

