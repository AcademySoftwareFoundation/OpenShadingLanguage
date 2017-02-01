Building OSL
============

OSL currently compiles and runs cleanly on Linux, Mac OS X, and Windows.

Dependencies
------------

OSL requires the following dependencies:

* [OpenImageIO](http://openimageio.org) 1.7 or newer

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
* [LLVM](http://www.llvm.org) 3.4, 3.5, 3.9, or 4.0

   It's possible that other intermediate versions will work, but we are not
   testing them. Please note that for version 3.5 or later, you'll need to
   be building OSL with C++11 (LLVM 3.4 is the last to support C++03).
   Currently, the newer versions (3.9 & 4.0) take longer to JIT, but the
   JITed code runs faster, so you may wish to consider this tradeoff (faster
   JIT may be important for interactive applications). We anticipate that
   future OSL releases will improve JIT performance and then drop support
   for the older LLVM versions.
* [Boost](www.boost.org) 1.55 or newer.
* [Imath/OpenEXR](http://openexr.com/downloads.html)
* [Flex](https://github.com/westes/flex)
* [GNU Bison](https://www.gnu.org/software/bison/)


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
   in "dist/ARCH/doc".

5. After building (and setting your library path), you can run the
   test suite with:

        make test
