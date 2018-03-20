#!/bin/csh


set stof_cmd=`testshade --batched -g 4 4 str_stof > stof_out.txt` 

set stof_cmd_ref=`testshade -g 4 4 str_stof > stof_ref.txt` 

set stof_diff=`diff stof_ref.txt stof_out.txt`

echo "\n"
echo "*******************"
echo "string stof()"
echo "*******************"


if ("$stof_diff" == "") then 
	echo "PASS" 
else
	echo "FAIL"
endif

