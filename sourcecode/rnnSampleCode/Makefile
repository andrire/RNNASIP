# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
PULP_APP = test
PULP_APP_FC_SRCS = test.c
PULP_APP_HOST_SRCS = test.c
PULP_CFLAGS = -O3 -g -lmath  -mhwloopmin=2
PULP_ARCH_CFLAGS = -march=rv32Xpulpv -lm
include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
