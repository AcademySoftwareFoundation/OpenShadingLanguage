#!/bin/csh


set substr_cmd=`testshade --batched -g 4 4 str_substr > substr_out.txt` 

set substr_cmd_ref=`testshade -g 4 4 str_substr > substr_ref.txt` 

set substr_diff=`diff substr_ref.txt substr_out.txt`

echo "\n"
echo "*******************"
echo "string substr()"
echo "*******************"

if ("$substr_diff" == "") then 
	echo "PASS" 
else
	echo "FAIL"
endif

