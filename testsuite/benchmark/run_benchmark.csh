testshade --param op $1 --param repeat $2 --stats --vary_udxdy --vary_udxdy -t 1 -g 64 64 -od uint8 -o Cout out_$1.tif benchmark > $1.log
# find the Run statistic, remove the "s" for seconds, reformat to testname, runt time
grep "Run  :" $1.log | sed "s/s//" | sed "s/Run  :/$1,/" | tee -a run_log.csv