# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
PULP_APP = testKernel

PULP_APP_FC_SRCS = basicKernel.c
PULP_APP_FC_SRCS += testKernel.c
# PULP_APP_HOST_SRCS = testKernel.c
# -mhwloopmin=2
PULP_CFLAGS = -O3 -g -mhwloopmin=0 -I./  
#deactivate optimizations
###############PULP_CL_ARCH_CFLAGS = -march=rv32imc -mPE=8 -mFC=1 -D__riscv__
###############PULP_FC_ARCH_CFLAGS = -march=rv32imc -mPE=8 -mFC=1 -D__riscv__
###############PULP_ARCH_LDFLAGS = -march=rv32imc -mPE=8 -mFC=1 -D__riscv__


include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
