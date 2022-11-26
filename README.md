# rempc
Standalone model-predictive control (MPC) codes. Plain `C` solver with `MEX` interface (known to work in both `Octave` and `Matlab`). The `C` solver has complexity linear in the length of the horizon.

## Build/test Octave from shell
Run `./test-octave-headless.sh` to compile with `mkoctfile` and run a basic test program (assuming Linux or WSL with `Octave` installed). 

## Build from within Octave or Matlab
To rebuild the `MEX` program from within either `Matlab` or `Octave`, just type `build_qpmpclti2f` in the prompt.

## Python interface (under construction)
Requires `numpy`. Rebuild and check a user site version by calling `./remake-user.sh`.

## Standalone tests of solver components
```
mkdir testbuild
cd testbuild
cmake ..
make
ctest
```
