<!-- Copyright Contributors to the Open Shading Language project.
     SPDX-License-Identifier: BSD-3-Clause -->

# Shared code for the test renderers

`testshade` and `testrender` are separate examples of OSL integration and are
also used by the testsuite. Both need the render state and free-function
renderer services in this directory. Their OptiX paths also use the CUDA
renderer library in `cuda/`.

Each example compiles the shared sources into its own LLVM bitcode and PTX.
There is no library target to link: the generated code is embedded in each
executable, and each executable installs PTX under its existing filename.
