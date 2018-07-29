# CUDA precipitate-aging notes

Due to the paraboloid free energy representation, we can leverage the HiPerC project to
implement the equations of motion on the GPU using CUDA.

## Workflow

### `MMSP::generate()`

Initial conditions will be generated using the existing MMSP code, and written to a
compressed MMSP checkpoint.

### Initialization

1. `MMSP.main.cpp` will be adapted to use HiPerC, rather than adapting HiPerC to
use MMSP.
2. The initial condition will be read back into an MMSP::grid object.
3. The Laplacian kernel will be written into const cache on the GPU.
4. 12 device arrays will be allocated on the GPU:
- Old values
    1. `x_Cr`
    2. `x_Nb`
    3. `phi_del`
    4. `phi_lav`
    5. `x_gam_Cr`
    6. `x_gam_Nb`
    - New values
    1. `x_Cr`
    2. `x_Nb`
    3. `phi_del`
    4. `phi_lav`
    5. `x_gam_Cr`
    6. `x_gam_Nb`
    5. 6 identical host arrays will be allocated on the CPU.

### `MMSP::update()`

#### Before timestepping:

    1. Data will be read from the MMSP::grid into the host arrays.
    2. Data will be copied from the host to device arrays.

#### For each iteration:

    1. Boundary conditions will be applied on each of the "old" device arrays.
    2. Laplacian values will be computed and recorded in each of the "new" arrays.
    3. "New" fields values will be computed from the two arrays using the "old" and
    Laplacian data.
    4. Fictitious compositions will be computed and written into the "new" array.

#### After timestepping:

    1. Data will be copied from the 6 "new" device arrays into the host arrays.
    2. Data will be read from the arrays into the MMSP::grid object.
    3. The MMSP::grid object will be written a compressed MMSP checkpoint.

### Cleanup

    Arrays will be freed from the host and device once the last checkpoint is written.
