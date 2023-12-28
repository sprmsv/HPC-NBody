#!/bin/bash

#SBATCH --reservation=Course-math-454-final
#SBATCH --account=math-454
#SBATCH --time=00:20:00

# Create profiling outputs
srun src/nbody data/galaxy.txt $1
srun perf record -o perf.data --call-graph dwarf src/nbody data/galaxy.txt $1

# Write the report of gprof
echo "=====================================================================" >> $2
echo "============================== GPROF ================================" >> $2
echo "=====================================================================" >> $2
gprof src/nbody >> $2
echo "" >> $2
echo "" >> $2
rm gmon.out

# Write perf stat
echo "=====================================================================" >> $2
echo "===========================  PERF-STAT  =============================" >> $2
echo "=====================================================================" >> $2
perf stat -e L1-dcache-loads,L1-dcache-load-misses src/nbody data/galaxy.txt $1 >> $2
echo "" >> $2
echo "" >> $2

# Write the report of perf
echo "=====================================================================" >> $2
echo "==========================  PERF-REPORT  ============================" >> $2
echo "=====================================================================" >> $2
perf report -g -i perf.data >> $2
echo "" >> $2
echo "" >> $2
rm perf.data
