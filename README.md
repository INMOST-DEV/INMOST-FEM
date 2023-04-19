# AniFem++
AniFem++ is flexible C++ library for constructing finite element method discretizations evolved from [Ani3D/AniFem](https://sourceforge.net/projects/ani3d/)

The core of this library consists of methods for assembling element matrices and classes for constructing FE spaces. For the INMOST platform, an interface is provided for assembling the global matrix, but for using other platforms and frameworks, the responsibility for implementing this functionality is left to the user.

## Installation

The core of the library is self-sufficient and does not require the connection of additional libraries, however, additional libraries are used to demonstrate examples and speed up some functions

The library has the following optional dependencies:
- [Eigen3](https://gitlab.com/libeigen/eigen) to speed-up some computations especially when using rich spaces or high orders of quadratures
- [INMOST](https://github.com/INMOST-DEV/INMOST) to compile interfaces to INMOST platform
- [CASADI](https://github.com/casadi/casadi) to compile some examples
- [SUNDIALS: KINSOL](https://computing.llnl.gov/projects/sundials/sundials-software) to use KinSol nonlinear solver in some examples 

### Superbuild installation
The simplest way to install this package with all optional dependencies is to use the superbuild interface (Internet connection is required), in which the package itself will download and install the basic versions of all the necessary dependencies. In order to use this approach, you should set the cmake variables `DOWNLOAD_<dep_name>`

To build the package in this way use the following command:
```console
mkdir build; cd build
cmake \
    -DWITH_INMOST=ON -DDOWNLOAD_inmost=ON   \
    -DWITH_KINSOL=ON -DDOWNLOAD_sundials=ON \
    -DWITH_EIGEN=ON  -DDOWNLOAD_eigen3=ON   \
    -DWITH_CASADI=ON -DDOWNLOAD_casadi=ON   \
    ../ 
cmake --build .
```

### Custom installation
If optional dependencies are already installed or you want to use customized versions of dependencies, you can use the standard cmake configure method, where the paths for dependencies are set explicitly

To build the package in this way use the following command:
```console
mkdir build; cd build 
cmake \
    -DWITH_INMOST=ON -DINMOST_ROOT=/full/path/inmost/instdir \
    -DWITH_KINSOL=ON -DSUNDIALS_ROOT=/full/path/sundials/instdir \
    -DWITH_CASADI=ON -Dcasadi_ROOT=/full/path/casadi/instdir \
    -DWITH_EIGEN=ON -DEigen3_ROOT=/full/path/eigen/instdir \
    ../ 
cmake --build .
```

## Usage guide and examples
Usage guides and examples are avaliable in [russian](docs/rus/user_guide.md) and [english](docs/eng/user_guide.md) variants.