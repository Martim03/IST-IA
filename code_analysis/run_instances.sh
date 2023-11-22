#!/bin/bash

GREEN='\033[0;32m'  # ANSI escape sequence for green color
NC='\033[0m'        # ANSI escape sequence to reset color

# Clean previous profiling data
rm -f output.pstats

for i in {01..09}
do
   echo "Testing instance$i..."
   # Run Python script with cProfile, appending to output.pstats
   python3 -m cProfile -o output.pstats ../bimaru.py < ../instances-students/instance${i}.txt > ../instances-students/my_output.out 2>&1
   if [ $? -eq 0 ]; then
	   echo -e "${GREEN}OK!${NC}"
	   echo "------------------------------------------------------------"
	   # Extract the user time from the time command's output
   fi
done

