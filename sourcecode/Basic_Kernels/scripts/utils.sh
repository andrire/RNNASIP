#!/bin/bash 
# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
function init {
cd ../../vp && source ../vp/init_vega_COS.bash
cd ../sourcecode/Basic_Kernels && make conf platform=gvsoc
sed -i '/PROFILING/s/^\/\///g' config_profiling.h
# turn on all models
sed -i '/MODEL/s/^/\/\//g' config_profiling.h


}

function activate_opt {
# comment them out
sed -i '/PULP_CL_ARCH_CFLAGS/s/^/#/g' Makefile
sed -i '/PULP_FC_ARCH_CFLAGS/s/^/#/g' Makefile
sed -i '/PULP_ARCH_LDFLAGS/s/^/#/g' Makefile
}

function deactivate_opt {
sed -i '/PULP_CL_ARCH_CFLAGS/s/^#//g' Makefile
sed -i '/PULP_FC_ARCH_CFLAGS/s/^#//g' Makefile
sed -i '/PULP_ARCH_LDFLAGS/s/^#//g' Makefile
}
