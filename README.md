> Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri 
> This is the open sourced version of the IIS internal repo https://iis-git.ee.ethz.ch/andri/RNNASIP.git 


# RNN ASIP
This is the main repository for the RNN Extension for PULP and tzscale (Synopsys)

The project has the following sub-repos and dependencies:

- RNN Pulpissimo (https://iis-git.ee.ethz.ch/andri/rnn-pulpissimo.git): Pulpissimo extended by special instructions
- RNN RISC-V core (https://iis-git.ee.ethz.ch/andri/rnn-riscv.git): Includes the RISC-V core with the RNN Extensions.
- RISC-V toolchain with RNN-Extensions (https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/tree/renzo-isa)
- The PULP SDK and ISS Simulator (https://github.com/pulp-platform/pulp-sdk.git)

## Getting Started

First of all, clone this repo (`rnn-pulpissimo` or `pulpissimo/rnnext`) and follow these instructions.

Clone and build the custom riscv toolchain at `pulp-riscv-gnu-toolchain` branch `renzo-isa` or skip within ETHZ/IIS network.
```
# Check Dependencies first
# centos: sudo yum install autoconf automake libmpc-devel mpfr-devel gmp-devel gawk  bison flex texinfo patchutils gcc gcc-c++ zlib-devel
# ubuntu16: sudo apt-get install autoconf automake autotools-dev curl libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev

cd vp
git clone https://github.com/pulp-platform/pulp-riscv-gnu-toolchain.git
cd pulp-riscv-gnu-toolchain/
# checkout working setup
git checkout 5d39fedd658d81a3ea9765cf9bd03a445b292e4b
git submodule update --init --recursive
./configure --prefix=/opt/riscv --with-arch=rv32imc --with-cmodel=medlow --enable-multilib
make
cd ../../
```
# Setting up and building the SDK
Clone the `pulp-sdk.git` (check README for dependencies):

```
git clone https://github.com/pulp-platform/pulp-sdk.git
cd pulp-sdk
# checkout latest tested SDK commit
git checkout f5f0d18ba7b589e3998425ea265bcc327e4f033f
git pull origin master
# within the IIS network: export PULP_RISCV_GCC_TOOLCHAIN=/usr/scratch/larain5/haugoug/public/pulp_riscv_gcc_renzo.3/
export PULP_RISCV_GCC_TOOLCHAIN=<path to the folder containing the bin folder of the toolchain>
#export VSIM_PATH=<pulpissimo root folder>/sim 
source configs/pulpissimo_rnnext.sh
source configs/platform-gvsoc.sh 
make all
echo "export PULP_RISCV_GCC_TOOLCHAIN=$PULP_RISCV_GCC_TOOLCHAIN" >> sourceme.sh
```

## Inititalize after build
To initialize all variable later, run the following commands:
```
export PULP_PROJECT_HOME=/path/to/pulp-sdk-git-repo
source ${PULP_PROJECT_HOME}/configs/pulpissimo_rnnext.sh
source ${PULP_PROJECT_HOME}/configs/platfom-gvsoc.sh
source ${PULP_PROJECT_HOME}/sourceme.sh
```

### Testing the SDK

The SDK build should have installed the SDK under `pkg/sdk/dev`.

Once you want to use it to compile and run applications, you first need to setup the SDK by sourcing the *sourceme.sh* file which is inside the installation folder:

    $ source ${PULP_PROJECT_HOME}/sourceme.sh

After these steps, the SDK is ready to be used, you can have a look at section *Documentation* for more information.

For a quick hello test, you can get some examples here:

    $ git clone https://github.com/pulp-platform/pulp-rt-examples.git

Then you can go to the folder `pulp-rt-examples/hello` and execute:

    $make conf 
    $make clean all run

# Generating the RTL platform
```
git clone https://github.com/pulp-platform/pulpissimo.git pulp_platform_rnnext
echo "export VSIM_PATH=$(pwd)/sim" >> ${PULP_PROJECT_HOME}/sourceme.sh

```
## Repo Structure

- `docs/` Documentation related scripts and meeting protocols
- `sourcecode/` C implementation and tzscale implementation (see README)
- `sourcecode/Basic_Kernels` C implementation of ML/RNN kernels (see README for more details)
- `sourcecode/rnnSampleCode` Verification of the rnn Extensions (see README)
- `sourcecode/tzscale` tzscale with RNN extensions (see README)
