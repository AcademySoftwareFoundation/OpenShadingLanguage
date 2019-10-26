Building OSL
============

OSL currently compiles and runs cleanly on Linux, Mac OS X, and Windows.

Dependencies
------------

OSL requires the following dependencies or tools.
NEW or CHANGED dependencies since the last major release are **bold**.

* Build system: [CMake](https://cmake.org/) 3.2.2 or newer

* A suitable C++11 compiler to build OSL itself, which may be any of:
   - GCC 4.8.5 or newer (through gcc 8)
   - Clang 3.4 or newer (through clang 9)
   - Microsoft Visual Studio 2015 or newer
   - Intel C++ compiler icc version 13 (?) or newer

  OSL should compile also properly with C++14 or C++17, but they are not
  required.

* **[OpenImageIO](http://openimageio.org) 2.0 or newer**

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
    variable $OPENIMAGEIOHOME to point to the compiled distribution, as
    well as for $OPENIMAGEIOHOME/lib to be in your LD_LIBRARY_PATH (or
    DYLD_LIBRARY_PATH on OS X) and then OSL's build scripts will be able
    to find it.

* **[LLVM](http://www.llvm.org) 5.0, 6.0, 7.0, 8.0, or 9.0**

   Optionally, if Clang libraries are installed alongside LLVM, OSL will
   in most circumstances use Clang's internals for C-style preprocessing of
   OSL source. If not found, it will fall back on Boost Wave (but on many
   platforms, that requires that Boost has been built in C++11 mode).

* [Boost](www.boost.org) 1.55 or newer.
* [Ilmbase](http://openexr.com/downloads.html) 2.0 or newer
* [Flex](https://github.com/westes/flex) and
  [GNU Bison](https://www.gnu.org/software/bison/)
* [PugiXML](http://pugixml.org/)
* [Partio](https://www.disneyanimation.com/technology/partio.html) --
  optional, but if it is not found at build time, the OSL `pointcloud`
  functions will not be operative.



Build process
-------------

Here are the steps to check out, build, and test the OSL distribution:

0. Install and build dependencies.

1. Check out a copy of the source code from the Git repository:

        git clone https://github.com/imageworks/OpenShadingLanguage.git osl

2. Change to the distribution directory and 'make'

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

3. After compilation, you'll end up with a full OSL distribution in
   dist/ARCH, where ARCH is the architecture you are building on, one of
   "linux", "linux64", "macosx", "windows", or "windows64".

   Note: The default is to make an optimized "release" build.  If
   instead type 'make debug' at the top level, you will end up with
   a debug build (no optimization, full symbols) in "dist/ARCH.debug".

4. Add the "dist/ARCH/bin" to your $PATH, and "dist/ARCH/lib" to your
   $LD_LIBRAY_PATH (or $DYLD_LIBRARY_PATH on OS X), or copy the contents
   of those files to appropriate directories.  Public include files
   (those needed when building applications that incorporate OSL)
   can be found in "dist/ARCH/include", and documentation can be found
   in "dist/ARCH/share/doc".

5. After building (and setting your library path), you can run the
   test suite with:

        make test
