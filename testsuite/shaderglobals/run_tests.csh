oslc test.osl
testshade -t 1 -g 16 2 test > sout.txt
testshade --batched -t 1 -g 16 2 test > bout.txt
diff -s sout.txt bout.txt 