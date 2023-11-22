#!/bin/bash

GREEN='\033[0;32m'  # ANSI escape sequence for green color
NC='\033[0m'        # ANSI escape sequence to reset color

rm -f output.pstats

for i in {01..10}
do
   echo "Testing instance$i..."
   diff_output=$(time python3 bimaru.py < instances-students/instance${i}.txt > instances-students/my_output.out 2>&1)
   if [ $? -eq 0 ]; then
       diff_result=$(diff instances-students/my_output.out instances-students/instance${i}.out)
       if [ $? -eq 0 ]; then
           echo -e "${GREEN}OK!${NC}"
           echo "------------------------------------------------------------"
       else
           echo "Output differs from expected instance${i}.out:"
           echo "$diff_result"
           echo "------------------------------------------------------------"
       fi
   else
       echo "An error occurred while running the script."
       echo "------------------------------------------------------------"
   fi
done

