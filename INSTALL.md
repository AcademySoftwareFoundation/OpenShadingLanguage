<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright Contributors to the Open Shading Language Project. -->

Building OSL
============

OSL currently compiles and runs cleanly on Linux, Mac OS X, and Windows.

Dependencies
------------

OSL requires the following dependencies or tools.
NEW or CHANGED dependencies since the last major release are **bold**.

* Build system: [CMake](https://cmake.org/) 3.12 or newer (tested through 3.26)

* A suitable C++14 or C++17 compiler to build OSL itself, which may be any of:
   - GCC 6.1 or newer (tested through gcc 12.1)
   - Clang 3.4 or newer (tested through clang 16)
   - Microsoft Visual Studio 2017 or newer
   - Intel C++ compiler icc version 17 or newer or LLVM-based icx compiler
     version 2022 or newer.

* **[OpenImageIO](http://openimageio.org) 2.3 or newer** (tested through 2.4)

    OSL uses OIIO both for its texture mapping functionality as well as
    numerous utility classes.  If you are integrating OSL into an existing
    renderer, you may use your own favorite texturing system rather than
    OpenImageIO with a little minor surgery.  There are only a few places
    where OIIO texturing calls are made, and they could easily be bypassed.
    But it is probably not possible to remove OIIO completely as a
    dependency, since we so heavily rely on a number of other utility classes
    that it provides (for which there was no point reinventing redundantly
    for OSL).

    After building OpenImageIO, if you don't have it installed in a
    "standard" place (like /usr/include), you should set the environment
    variable $OpenImageIO_ROOT to point to the compiled distribution, and
    then OSL's build scripts will be able to find it. You should also have
    $OpenImageIO_ROOT/lib to be in your LD_LIBRARY_PATH (or
    DYLD_LIBRARY_PATH on OS X).

* [LLVM](http://www.llvm.org) 9, 10, 11, 12, 13, 14, or 15, including
  clang libraries. LLVM 16 doesn't work yet, we need to make changes
  on the OSL side to be compatible.

* (optional) For GPU rendering on NVIDIA GPUs:
    * [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) 7.0 or higher.
    * [Cuda](https://developer.nvidia.com/cuda-downloads) 8.0 or higher.

* [Boost](https://www.boost.org) 1.55 or newer (tested through boost 1.81)
* [Ilmbase or Imath](https://github.com/AcademySoftwareFoundation/openexr) 2.3
   or newer (recommended: 2.4 or higher; tested through 3.1)
* [Flex](https://github.com/westes/flex) 2.5.35 or newer and
  [GNU Bison](https://www.gnu.org/software/bison/) 2.7 or newer.
  Note that on some MacOS/xcode releases, the system-installed Bison is too
  old, and it's better to install a newer Bison (via Homebrew is one way to
  do this easily).
* [PugiXML](http://pugixml.org/) >= 1.8 (we have tested through 1.13).
* (optional) [Partio](https://www.disneyanimation.com/technology/partio.html)
  If it is not found at build time, the OSL `pointcloud` functions will not
  be operative.
* (optional) Python: If you are building the Python bindings or running the
  testsuite:
    * Python >= 2.7 (tested against 2.7, 3.7, 3.8, 3.9, 3.10)
    * pybind11 >= 2.4.2 (Tested through 2.10)
    * NumPy
* (optional) Qt5 >= 5.6 or Qt6 (tested Qt5 through 5.15 and Qt6 through 6.4).
  If not found at build time, the `osltoy` application will be disabled.



Build process
-------------

Here are the steps to check out, build, and test the OSL distribution:

1. Install and build dependencies.

2. Check out a copy of the source code from the Git repository:

        git clone https://github.com/AcademySoftwareFoundation/OpenShadingLanguage.git osl

3. Change to the distribution directory and 'make'

        cd osl
        make

   Note: OSL uses 'CMake' for its cross-platform build system.  But for
   simplicity, we have made a "make wrapper" around it, so that by just
   typing 'make' everything will build.  Type 'make help' for other 
   options, and note that 'make nuke' will blow everything away for the
   freshest possible compile.

   NOTE: If the build breaks due to compiler warnings which have been
   elevated to errors, you can try "make clean" followed by
   "make STOP_ON_WARNING=0", that create a build that will only stop for
   full errors, not warnings.

4. After compilation, you'll end up with a full OSL distribution in
   dist/

5. Add the "dist/bin" to your $PATH, and "dist/lib" to your
   $LD_LIBRAY_PATH (or $DYLD_LIBRARY_PATH on MacOS), or copy the contents
   of those files to appropriate directories.  Public include files
   (those needed when building applications that incorporate OSL)
   can be found in "dist/include", and documentation can be found
   in "dist/share/doc".

6. After building (and setting your library path), you can run the
   test suite with:

        make test
        
Troubleshooting
----------------

- [Build issues on macOS Catalina (fatal error: 'wchar.h' file not found)](https://github.com/AcademySoftwareFoundation/OpenShadingLanguage/issues/1055#issuecomment-581920327)
