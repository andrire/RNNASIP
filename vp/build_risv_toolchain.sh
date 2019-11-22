#!/bin/bash
# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri

# Dependencies
# centos: sudo yum install autoconf automake libmpc-devel mpfr-devel gmp-devel gawk  bison flex texinfo patchutils gcc gcc-c++ zlib-devel
# ubuntu16: sudo apt-get install autoconf automake autotools-dev curl libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev

export postfix="CentOS"

git clone --recursive https://github.com/pulp-platform/pulp-riscv-gnu-toolchain pulp-riscv-gnu-toolchain_$postfix

cd pulp-riscv-gnu-toolchain_$postfix
git checkout isa-renzo

echo "Make PULP"
./configure --prefix=$(pwd)/opt-riscv/ --with-arch=rv32imc --with-cmodel=medlow --enable-multilib
make 
cd ../




