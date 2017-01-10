Building OSL
============

OSL currently compiles and runs cleanly on Linux and Mac OS X.  We have
not yet compiled it for Windows, but we believe we it should be very
portable (we have done almost nothing platform-specific).

OSL makes very heavy use of the OpenImageIO project
(http://openimageio.org), both for its texture mapping functionality as
well as numerous utility classes.  If you are integrating OSL into an
existing renderer, you may use your own favorite texturing system rather
than OpenImageIO with a little minor surgery.  There are only a few
places where OIIO (OpenImageIO) texturing calls are made, and they could
easily be bypassed.  But it is probably not possible to remove OIIO
completely as a dependency, since we so heavily rely on a number of
other utility classes that it provides (for which there was no point
reinventing redundantly for OSL).

Here are the steps to check out, build, and test the OSL distribution:

0. Install and build dependencies.  You will need Boost (www.boost.org),
   Imath (http://openexr.com/downloads.html), and OpenImageIO
   (http://openimageio.org).

   After building OpenImageIO, if you don't have it installed in a
   "standard" place (like /usr/include), you should set the environment
   variable $OPENIMAGEIOHOME to point to the compiled distribution, as
   well as for $OPENIMAGEIOHOME/lib to be in your LD_LIBRARY_PATH (or
   DYLD_LIBRARY_PATH on OS X) and then OSL's build scripts will be able
   to find it.

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

   (Note: currently all tests pass on OS X but a few tests fail on Linux
   strictly for floating point precision reasons, not because anything
   is really broken.  We're working on a fix for this.)
