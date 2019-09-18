#!/bin/sh
# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri
#######################################################################################################################
## @brief Clean run (helper script)
## @file cleanrun.sh 
##
## @author Renzo Andri (andrire)
##

rm -r build/*
make clean
make all run
