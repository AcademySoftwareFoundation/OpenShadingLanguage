#!/bin/csh

oslc test_error.osl
oslc test_warning.osl

echo "\n"
echo "*******************"
echo "osl_error"
echo "*******************"

set error_cmd=`testshade --batched -g 200 200 test_error  > error_out.txt` 

set error_cmd_ref=`testshade -g 200 200 test_error  > error_ref.txt` 

set error_diff=`diff error_ref.txt error_out.txt`



if ("$error_diff" == "") then 
    echo "PASS" 
else
    echo "FAIL"
endif


echo "\n"
echo "*******************"
echo "osl_warning"
echo "*******************"

set warning_cmd=`testshade --batched -g 200 200 test_warning  > warning_out.txt` 

set warning_cmd_ref=`testshade -g 200 200 test_warning  > warning_ref.txt` 

set warning_diff=`diff warning_ref.txt warning_out.txt`



if ("$warning_diff" == "") then 
    echo "PASS" 
else
    echo "FAIL"
endif


