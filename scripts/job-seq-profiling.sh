#!/bin/bash

#SBATCH --reservation=Course-math-454-final
#SBATCH --account=math-454
#SBATCH --time=00:20:00

METHOD=$1
OUT=$2

# Create profiling outputs
srun src/seq/nbody data/galaxy.txt 10 $METHOD
srun perf record -o perf.data --call-graph dwarf src/seq/nbody data/galaxy.txt 10 $METHOD

# Write the report of gprof
echo "=====================================================================" >> $OUT
echo "============================== GPROF ================================" >> $OUT
echo "=====================================================================" >> $OUT
gprof src/seq/nbody >> $OUT
echo "" >> $OUT
echo "" >> $OUT
rm gmon.out

# Write perf stat
echo "=====================================================================" >> $OUT
echo "===========================  PERF-STAT  =============================" >> $OUT
echo "=====================================================================" >> $OUT
perf stat -e L1-dcache-loads,L1-dcache-load-misses src/seq/nbody data/galaxy.txt $METHOD >> $OUT
echo "" >> $OUT
echo "" >> $OUT

# Write the report of perf
echo "=====================================================================" >> $OUT
echo "==========================  PERF-REPORT  ============================" >> $OUT
echo "=====================================================================" >> $OUT
perf report -g -i perf.data >> $OUT
echo "" >> $OUT
echo "" >> $OUT
rm perf.data
