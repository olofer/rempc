# rempc
Standalone model-predictive control (MPC) codes. Includes a library-free plain `C` solver with (i) `MEX` interface (known to work in both `Octave` and `Matlab`) and (ii) `Python/numpy` interface. The `C` solver has complexity linear in the length of the horizon.

## Build/check Python interface (not fully tested)
Requires `numpy` (tests also require `scipy`). Run the following.
```
python3 setup.py install --user
python3 test.py --smoke
```
(it is also possible to rebuild and check the user site version in a single go by calling `./remake-user.sh`).

See script `test.py` for example usage of the basic MPC code. Specifically the code executed under the `--tripleint` option. The solver performance can be visualized by running e.g. `python3 test.py --profile-solver --tripleint --horizon 250`. The plot (generated PDF) should show the wall clock required to solve the MPC program as a function of the number of time steps in the horizon.

### Jupyter notebook demonstrations
- `test-mpc-afti16-slack.ipynb` (feedback control of unstable $2\times 2$ plant with hard and soft constraints)

### Not yet implemented
- Interface for LTV problems
- Interface for MHE problems

## Build/test Octave from shell
Run `./test-octave-headless.sh` to compile with `mkoctfile` and run a basic test program (assuming Linux or WSL with `Octave` installed). 

## Build/demo from within Octave or Matlab
To rebuild the `MEX` program from within either `Matlab` or `Octave`, just type `build_qpmpclti2f` in the prompt. The scripts `test_mpc_*.m` contains demonstrations and tests of the solver. In `Octave` you may need to run `pkg load control` before running the demos. In `Matlab` you may need to have the control systems toolbox. The "reference" program `qpmpclti2e.m` is largely equivalent to the `C` solver `qpmpclti2f.c` (but much slower).

## Standalone tests of solver components
```
mkdir testbuild
cd testbuild
cmake ..
make
ctest
```
