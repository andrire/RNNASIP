#!/bin/bash
# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri

# Settings
OUTPUTBUFFER_SWEEP="true"
CREATE_STATISTICS="true"
RUN_AND_CHECK_CORRECTNESS="true"
#gvsoc or rtl
PLATFORM="gvsoc"
INPUTFMTILING="false"
PREFIX="VLIW_prefetch"

#Change platform
if [ "$PLATFORM" = rtl ]; then
source $PULP_PROJECT_HOME/configs/platform-rtl.sh
echo "Simulation platform is here: VSIM_PATH=${VSIM_PATH}"
#get around relative path from rtl platform
MAKE_FLAGS="vsim/script=../../../../../../../../../../../../../$(pwd)/scripts/vsim_run_and_exit.tcl"
MSECOL=4
else
source $PULP_PROJECT_HOME/configs/platform-gvsoc.sh
MAKE_FLAGS=""
MSECOL=2
fi

mse_criteria=100 #float to fixed point comparison has some error, acceptable error

function turnOn {
sed -i "s/.*$1\.*/#define $1/" config_profiling.h
}
function turnOff {
sed -i "s/.*$1\.*/\/\/ #define $1/" config_profiling.h
}
function setValue {
sed -i "s/.*$1\.*/\/\/ #define $1 $2/" config_profiling.h
}




#models="MODEL2 MODEL5 MODEL6 MODEL7 MODEL8 MODEL10 MODEL11 MODEL9"
models="MODEL0 MODEL1 MODEL2 MODEL3 MODEL5 MODEL6 MODEL7 MODEL8 MODEL9 MODEL10 "

if [ "$INPUTFMTILING" = "both" ]; then 
inputfmtiling="false true"
else
inputfmtiling="$INPUTFMTILING"
fi

if [ "$OUTPUTBUFFER_SWEEP" = true ]; then 
outputbuffer="2 1 4 6 8 10 12"
else
#take same as in config
export outputbuffer=$(cat config_profiling.h | grep OUTPUTBUFFER | cut -d" " -f3)
fi

for numbuff in $outputbuffer; do
for input_tiling in $inputfmtiling; do
for i in $models; do
echo "" > config_profiling.h
  for j in $models; do
     echo "//#define $j" >> config_profiling.h
  done

echo "#define $i" >> config_profiling.h
echo "#define OUTPUTBUFFER $numbuff" >> config_profiling.h
if [ "$input_tiling" = true ]; then
echo "#define FMINTILING" >> config_profiling.h
else
echo "//#define FMINTILING" >> config_profiling.h
fi
if [ "$numbuff" != "1" ]; then
echo "#define FMOUTTILING" >> config_profiling.h
else
echo "//#define FMOUTTILING" >> config_profiling.h
fi

testName="${PREFIX}_${i}_obuff${numbuff}_${input_tiling}_$PLATFORM"
export  TRACE_FILE="reports/${testName}.log"
echo "" > $TRACE_FILE
if [ "$CREATE_STATISTICS" = true ]; then
source scripts/run_insn_statistic.sh  | grep sum
fi
if [ "$RUN_AND_CHECK_CORRECTNESS" = true ]; then
   export  TRACE_FILE="reports/${testName}.log2"
   make clean all run $MAKE_FLAGS >>  $TRACE_FILE 2>&1
   # echo $TRACE_FILE
   mse=$(cat $TRACE_FILE | grep mse | head -n1 | cut -d" " -f$MSECOL )
   # echo -e "asdf [\e[1;31failed\e[0m]"
   if [ -z "$mse" ]; then
      
      if grep -q "collect2: error: ld returned 1 exit status" $TRACE_FILE; then
       printf "%20s   \t[\e[1;31mfailed \e[0m] (due to memory constraints)\n" $testName
      else
       printf "%20s   \t[\e[1;31mfailed \e[0m] (due to unknown reason)\n" $testName
      fi
   else
      if [ $mse -lt $mse_criteria ]; then
      printf "%30s \t[\e[0;32msuccess\e[0m] (%4d)\n" $testName $mse
      else
         printf "%30s \t[\e[1;31mfailed \e[0m] (%4d)\n" $testName $mse
      fi
   fi
fi
done
done
done

# static inline void print_result(char* test, int actual, int expected) {
# printf("%-80.80s", test); printf(":\t");
# if(expected == actual)
# printf("%8.8x vs. Exp. %8.8x [\033[0;32msuccess\033[0m] \n", actual, expected);
# else
# printf("%8.8x vs. Exp. %8.8x [\033[1;31mfailed\033[0m] \n", actual, expected);
# }
