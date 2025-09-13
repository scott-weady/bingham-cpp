# Bingham closure

This repository contains a C++ implementation of the 3D apolar Doi-Saintillan-Shelley kinetic theory with the Bingham closure. Parameters, including dimensionless parameters and spatial and temporal resolutions, should be specified in the `params.toml` file. The code can be compiled and run using the following commands

```
make
build/main <number of OpenMP threads> <0 to restart or 1 to include initial data>
```

By default, it will use the maximum number of threads available and will generate a new initial condition. Initial data can be provided in a file named `initial_data/Q.dat`, and must be compatible with the specified resolution. The code will output, in the results directory, low resolution files of `Qmag` (scalar nematic order parameter) and `umag` (velocity magnitude) for plotting and high resolution files of `Q` (upper triangular components of the nematic tensor) and `u` (velocity field). The frequency can be specificed in `params.toml`.

## Dependencies

Requires fftw and C++17 (for `toml.hpp`).
