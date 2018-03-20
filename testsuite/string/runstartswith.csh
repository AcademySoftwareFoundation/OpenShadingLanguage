#!/bin/csh


set startswith_cmd=`testshade --batched -g 4 4 str_startswith > sw_out.txt` 

set startswith_cmd_ref=`testshade -g 4 4 str_startswith > sw_ref.txt` 

set sw_diff=`diff sw_ref.txt sw_out.txt`

echo "\n"
echo "*******************"
echo "string startswith()"
echo "*******************"


if ("$sw_diff" == "") then 
	echo "PASS" 
else
	echo "FAIL"
endif

