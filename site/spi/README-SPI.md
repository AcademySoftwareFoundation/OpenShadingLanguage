Build a variant for Rez/general use
-----------------------------------

Skip this section if you are instead building for SpComp2.

Note: For testing or to make your own local rez package, you can customize
either the rez package name, or the rez install location name, with make
flags `OSL_REZ_NAME=blah REZ_PACKAGE_ROOT=/path/to/my/rez/pkgs` appended
to the `make` commands of any of the variants listed below. For example,

    make OSL_SPIREZ=1 OSL_REZ_NAME=oiio_test REZ_PACKAGE_ROOT=/path/to/my/rez/pkgs


Variants:

    # C++11/gcc4.8 compat, Python 2.7, Boost 1.55, OptiX 6, Cuda 10.1
    # OIIO 2.1.5.1, OpenEXR 2.2
    make nuke
    make spi OSL_SPIREZ=1 BOOSTVERS=1.55 USE_OPTIX=1 CUDA_VERSION=10.1 OPTIX_VERSION=6.0.0 OPENIMAGEIO_VERSION=2.1.5.1 OPENEXR_VERSION=2.2.0 PYTHON_VERSION=2.7



You can do any of these on your local machine.


Rez/general release (do for each variant)
-----------------------------------------

This must be done from compile42 (for correct write permissions on certain
shared directories), even if you did the build itself locally.

For any of the variants that you built above:

    ( cd dist/rhel7 ; rez release --skip-repo-errors )

That command will release the dist to the studio.


Appwrapper binary releases
--------------------------

This step is for the ONE general/rez variant that we believe is the
canonical source of command line oiiotool and maketx. After building and
releasing as above,

    cp dist/rhel7/OSL_*.xml /shots/spi/home/lib/app_cfg/OSL

That will make appcfg aware of the release.

To also make this release the new facility default:

    db-any spi/home/OSL.bin highest /shots/spi/home/lib/app_cfg/OSL/OSL_A.B.C.D.xml

where A.B.C.D is the version.


SpComp2 build and release
-------------------------

TBD
