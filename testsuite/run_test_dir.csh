#!/bin/csh

cd $1 
printf  "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
printf  "<<<<   ./$1 \n"
./run_tests.csh >& test.log
set err_count=`grep ERROR test.log | wc -l`
set fail_count=`grep FAIL test.log | wc -l`
set pass_count=`grep PASS test.log | wc -l`
set warning_count=`grep WARNING test.log | wc -l`
#printf "Errors= $err_count Fails= $fail_count Passed=$pass_count \n"
printf "<<<<   $pass_count Passed"
printf "\n"
if ( $warning_count != "0" ) then
    printf "<<<<   $warning_count WARNINGS(s)\n"
endif
if ( $err_count != "0" ) then
    printf "<<<<   $err_count ERROR(s)\n"
endif
if ( $2 != "" ) then
    printf "<<<<   $2 ERROR(s) expected\n"
endif
if ( $fail_count != "0" ) then
    printf "<<<<   $fail_count FAILURE(s)\n"
endif
if ( ($err_count != 0 && $err_count != $2) || $fail_count != "0") then
    printf "<<<<   cat test.log:\n"
    cat test.log
endif
#printf  ">>>>   end tests in '$1' \n"
printf  "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n"

cd ..
