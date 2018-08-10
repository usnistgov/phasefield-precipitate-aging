# CUDA precipitate-aging notes

Due to the paraboloid free energy representation, we can leverage the
[HiPerC](https://github.com/usnistgov/hiperc) project to implement the
equations of motion on the GPU using CUDA.

## Workflow

### `MMSP::generate()`

Initial conditions are generated using the existing MMSP code, and written to a
compressed MMSP checkpoint, with no work assigned to the GPU.

### Initialization

1. `main.cpp` was adapted from MMSP, rather than HiPerC.
2. The initial condition checkpoint gets read back into an MMSP::grid object.
3. The Laplacian kernel gets written into const cache on the GPU.
4. 12 device arrays get allocated on the GPU: one old and one new for each field variable.
    1. `x_Cr`
    2. `x_Nb`
    3. `phi_del`
    4. `phi_lav`
    5. `x_gam_Cr`
    6. `x_gam_Nb`
5. 12 identical host arrays get allocated on the CPU.

### `MMSP::update()`

#### Before timestepping:

1. Data gets read from the MMSP::grid into the host arrays.
2. Data gets copied from the host to device arrays.

#### For each iteration:

1. Boundary conditions get applied on each of the "old" device arrays.
2. Laplacian values gets computed and recorded in each of the "new" arrays.
3. Boundary conditions get applied on each of the "new" device arrays.
4. Updated field values get computed from the "old" and "new" device arrays.
5. Fictitious matrix phase compositions get computed and written into the "new" device  array.

#### After timestepping:

1. Data gets copied from the 6 "new" device arrays into the corresponding host arrays.
2. Data gets read from the host arrays into the MMSP::grid object.
3. The MMSP::grid object gets written to a compressed MMSP checkpoint.

### Cleanup

Arrays gets freed from the host and device once the last checkpoint is written.
