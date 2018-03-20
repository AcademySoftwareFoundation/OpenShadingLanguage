#!/bin/csh


set endswith_cmd=`testshade --batched -g 4 4 str_endswith > ew_out.txt` 

set endswith_cmd_ref=`testshade -g 4 4 str_endswith > ew_ref.txt` 

set ew_diff=`diff ew_ref.txt ew_out.txt`

echo "\n"
echo "*******************"
echo "string endswith()"
echo "*******************"


if ("$ew_diff" == "") then 
	echo "PASS" 
else
	echo "FAIL"
endif

