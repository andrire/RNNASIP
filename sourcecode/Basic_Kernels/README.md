Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri

The main machine learning kernels for this project can be found in this folder (i.e. /RNNASIP/sourcecode/Basic_Kernels).

## Setup the Toolchain
Set up tool chain:
```
previous_pwd = $(pwd)
cd $PATH_TO_PULP_SDK
export PULP_RISCV_GCC_TOOLCHAIN=$PATH_TO_RISCV_GCC_TOOLCHAIN
source configs/pulpissimo_rnnext.sh
source pkg/sdk/dev/sourceme.sh 
PULP_CONFIGS_PATH=`pwd`/pulp-configs/configs/:`pwd`/pkg/sdk/dev/install/ws/configs
cd $previous_pwd
make conf
```

## Create Benchmarks
All Networks have been defined in ```scripts/BenchmarkNetworks.py```. If needed add a new network or export the networks to create the ```benchmarks.h``` header file.

```
python3 scripts/BenchmarkNetworks.py
```

## Run the network on the SDK: 
```
make all run
```
Tip: ```make clean``` does not always work properly, use ```rm -rf build && make clean all run```.

## Run the network with traces:
With the CONFIG_OPT attribute, it can be defined which traces should be shown (e.g. insn for all instructions)
```
make all run CONFIG_OPT=gvsoc/trace=insn
```

## Run on the RTL platform
Dependencies: RTL platform set up and build (see corresponding README for details)
```
export VSIM_PATH=<path to the RTL platform>/sim
source $PULP_PROJECT_HOME/configs/platform-rtl.sh
make clean all run gui=1
```

## Run gate-level simulation and create vcd file for power simulation
```
export APP_PATH=$(pwd)
cd $VSIM_PATH
ln -s <path-to-gate-level-netlist> gate_level_netlist.v
make gate_all
cd $APP_PATH
make clean all run gui=1
```

## Run instruction analysis and profiling
```
# run statistics for active network
bash scripts/run_insn_statistic.sh
# run profiling for all blocks and models
python3 scripts/profiling_loop.py
```

## Run verification suite
The verification can be run with the ```run_benchmark.sh``` script. The following settings can be adapted:<br/>
```
OUTPUTBUFFER_SWEEP="true"           # sweeps over all the different output FM tile sizes.
CREATE_STATISTICS="false"           # run instruction settings as well
RUN_AND_CHECK_CORRECTNESS="true"    # verifiy correctness
PLATFORM="rtl"                      # rtl|gvsoc
INPUTFMTILING="both"                # true|false|both
```

```
source run_benchmark.sh
```
