#!/bin/csh


set hash_cmd=`testshade --batched -g 4 4 str_hash > hash_out.txt` 

set hash_cmd_ref=`testshade -g 4 4 str_hash > hash_ref.txt` 

set hash_diff=`diff hash_ref.txt hash_out.txt`

echo "\n"
echo "*******************"
echo "string hash()"
echo "*******************"

if ("$hash_diff" == "") then 
	echo "PASS" 
else
	echo "FAIL"
endif

