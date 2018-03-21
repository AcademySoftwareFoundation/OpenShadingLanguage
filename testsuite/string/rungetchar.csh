#!/bin/csh


set getchar_cmd=`testshade --batched -g 4 4 str_getchar > getchar_out.txt`

set getchar_cmd_ref=`testshade -g 4 4 str_getchar > getchar_ref.txt` 

set getchar_diff=`diff getchar_ref.txt getchar_out.txt`

echo "\n"
echo "*******************"
echo "string getchar()"
echo "*******************"

#echo "Value of hash_diff\n"
#echo $hash_diff

if ("$getchar_diff" == "") then 
	echo "PASS" 
else
	echo "FAIL"
endif

