#!/bin/bash
# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri

if [ -z ${TRACE_FILE+x} ]; then TRACE_FILE=reports/trace_`date "+%F-%T"`; fi
if [ -z ${TRACE_OPT+x} ]; then TRACE_OPT=gvsoc/trace=insn; fi
make clean all run CONFIG_OPT="${TRACE_OPT}" &> ${TRACE_FILE}
cat ${TRACE_FILE} | grep -E "insn|Start|\n"  > tmp
echo "Created traces and stored in ${TRACE_FILE}"

python3 scripts/create_statistic.py tmp Start | tee ${TRACE_FILE}_summary
echo "Created instruction summary in ${TRACE_FILE}_summary"
rm tmp
# CONFIG_OPT=gvsoc/trace=insn
