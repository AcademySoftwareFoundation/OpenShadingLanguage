#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

echo "gh-win-installdeps.bash"
env | sort

# DEP_DIR="$PWD/ext/dist"
DEP_DIR="$PWD/dist"
mkdir -p "$DEP_DIR"
mkdir -p ext
VCPKG_INSTALLATION_ROOT=/c/vcpkg

# Retry downloads a few times to ride out transient network failures, and
# fail on HTTP errors rather than saving an error page as the download.
CURL_RETRY_FLAGS="--fail --retry 5 --retry-delay 10"

export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:=.}
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;$DEP_DIR"
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;$VCPKG_INSTALLATION_ROOT/installed/x64-windows-release"
export PATH="$PATH:$DEP_DIR/bin:$DEP_DIR/lib:$VCPKG_INSTALLATION_ROOT/installed/x64-windows-release/bin:$PWD/ext/dist/bin:$PWD/ext/dist/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$DEP_DIR/bin:$VCPKG_INSTALLATION_ROOT/installed/x64-windows-release/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$DEP_DIR/lib:$VCPKG_INSTALLATION_ROOT/installed/x64-windows-release/lib"


if [[ "$PYTHON_VERSION" == "3.9" ]] ; then
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;/c/hostedtoolcache/windows/Python/3.9.13/x64"
    export Python_EXECUTABLE="/c/hostedtoolcache/windows/Python/3.9.13/x64/python3.exe"
    export PYTHONPATH=$OpenImageIO_ROOT/lib/python${PYTHON_VERSION}/site-packages
elif [[ "$PYTHON_VERSION" == "3.12" ]] ; then
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;/c/hostedtoolcache/windows/Python/3.12.10/x64"
    export Python_EXECUTABLE="/c/hostedtoolcache/windows/Python/3.12.10/x64/python3.exe"
    export PYTHONPATH=$OpenImageIO_ROOT/lib/python${PYTHON_VERSION}/site-packages
elif [[ "$PYTHON_VERSION" == "3.14" ]] ; then
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;/c/hostedtoolcache/windows/Python/3.14.3/x64"
    export Python_EXECUTABLE="/c/hostedtoolcache/windows/Python/3.14.3/x64/python3.exe"
    export PYTHONPATH=$OpenImageIO_ROOT/lib/python${PYTHON_VERSION}/site-packages
fi
pip install numpy


########################################################################
# Dependency method #1: Use vcpkg (disabled)
#
# Currently we are not using this, but here it is for reference:
#
echo "All pre-installed VCPkg installs:"
vcpkg list
echo "---------------"
# vcpkg update
# 
# Example of how it's done:
# vcpkg install tiff:x64-windows-release
# vcpkg install pugixml:x64-windows-release
# vcpkg install opencolorio:x64-windows-release
# vcpkg install openimageio:x64-windows-release

#echo "$VCPKG_INSTALLATION_ROOT"
#ls -R "$VCPKG_INSTALLATION_ROOT" || true
# 
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$VCPKG_INSTALLATION_ROOT/installed/x64-windows-release"
# 
echo "All VCPkg installs:"
vcpkg list
#
########################################################################


########################################################################
# Dependency method #2: Build from source ourselves
#
# If the CI workflow restored the built dependencies from its cache (see
# gh-win-deps-cache-key.bash), OSL_DEPS_CACHE_HIT is "true" and we skip
# the builds, only setting the variables that the build scripts export.
#

# build_dep SCRIPT ROOTVAR : source a dependency build script, or on a
# cache hit, just set ROOTVAR to where the build would have installed it.
build_dep () {
    if [[ "$OSL_DEPS_CACHE_HIT" == "true" ]] ; then
        echo "Using cached build from $1"
        export $2=$LOCAL_DEPS_DIR/dist
    else
        source src/build-scripts/$1
    fi
}

if [[ "$OSL_DEPS_CACHE_HIT" != "true" ]] ; then
    src/build-scripts/build_zlib.bash
    src/build-scripts/build_libpng.bash
fi
export ZLIB_ROOT=$PWD/ext/dist
export PNG_ROOT=$PWD/ext/dist

build_dep build_pybind11.bash pybind11_ROOT
export pybind11_ROOT=$PWD/ext/dist


if [[ "$OPENEXR_VERSION" != "" ]] ; then
    build_dep build_openexr.bash OpenEXR_ROOT
fi

if [[ "$PUGIXML_VERSION" != "" ]] ; then
    build_dep build_pugixml.bash pugixml_ROOT
    export OSL_CMAKE_FLAGS+=" -DUSE_EXTERNAL_PUGIXML=1 "
fi

if [[ "$OPENCOLORIO_VERSION" != "" ]] ; then
    build_dep build_opencolorio.bash OpenColorIO_ROOT
fi

cp $DEP_DIR/lib/*.lib $DEP_DIR/bin || true
cp $DEP_DIR/bin/*.dll $DEP_DIR/lib || true

if [[ "$OPENIMAGEIO_VERSION" != "" ]] ; then
    # There are many parts of OIIO we don't need to build for OSL's tests
    export ENABLE_iinfo=0 ENABLE_iv=0 ENABLE_igrep=0
    export ENABLE_iconvert=0 ENABLE_testtex=0
    # For speed of compiling OIIO, disable the file formats that we don't
    # need for OSL's tests
    export ENABLE_BMP=0 ENABLE_cineon=0 ENABLE_DDS=0 ENABLE_DPX=0 ENABLE_FITS=0
    export ENABLE_ICO=0 ENABLE_iff=0 ENABLE_jpeg2000=0 ENABLE_PNM=0 ENABLE_PSD=0
    export ENABLE_RLA=0 ENABLE_SGI=0 ENABLE_SOCKET=0 ENABLE_SOFTIMAGE=0
    export ENABLE_TARGA=0 ENABLE_WEBP=0 ENABLE_jpegxl=0 ENABLE_libuhdr=0
    # We don't need to run OIIO's tests
    export OPENIMAGEIO_CMAKE_FLAGS+=" -DOIIO_BUILD_TESTING=OFF -DOIIO_BUILD_TESTS=0"
    # Don't let warnings in OIIO break OSL's CI run
    export OPENIMAGEIO_CMAKE_FLAGS+=" -DSTOP_ON_WARNING=OFF"
    # export OPENIMAGEIO_CMAKE_FLAGS+=" -DUSE_OPENGL=0"
    export USE_QT=0 USE_OPENCV=0 USE_FFMPEG=0 USE_QT=0
    if [[ "${OPENIMAGEIO_UNITY:-1}" != "0" ]] ; then
        # Speed up the OIIO build by doing a "unity" build. (Note: this is
        # only a savings in CI where there are only 1-2 cores available.)
        export OPENIMAGEIO_CMAKE_FLAGS+=" -DCMAKE_UNITY_BUILD=ON -DCMAKE_UNITY_BUILD_MODE=BATCH"
    fi
    build_dep build_openimageio.bash OpenImageIO_ROOT
    export PYTHONPATH=$OpenImageIO_ROOT/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH
fi

cp $DEP_DIR/lib/*.lib $DEP_DIR/bin || true
cp $DEP_DIR/bin/*.dll $DEP_DIR/lib || true
echo "DEP_DIR $DEP_DIR :"
ls -R -l "$DEP_DIR"

if [[ "$LLVM_VERSION" != "" ]] ; then
    echo "Running llvm_build.bash"
    source src/build-scripts/build_llvm.bash
elif [[ "$LLVM_GOOGLE_DRIVE_ID" != "" ]] ; then
    mkdir -p $HOME/llvm
    pushd $HOME/llvm
    # The llvm-build dir may already be present if restored by actions/cache
    # (keyed on LLVM_GOOGLE_DRIVE_ID), in which case skip the download. This
    # also keeps us from tripping Google Drive's per-file download quota.
    if [[ -d llvm-build ]] ; then
        echo "Using cached LLVM from google drive id ${LLVM_GOOGLE_DRIVE_ID}"
    else
        echo "Installing LLVM from google drive"
        #LLVM_GOOGLE_DRIVE_ID="1uy7PNVlTQ-H56unXGOS6siRWtNcdS1J7"
        LLVM_ZIP_FILENAME=llvm-build.zip
        time curl $CURL_RETRY_FLAGS --insecure -L "https://drive.usercontent.google.com/download?id=${LLVM_GOOGLE_DRIVE_ID}&confirm=xxx" -o $LLVM_ZIP_FILENAME
        # When the download quota is exceeded, Google Drive returns an HTML
        # error page with HTTP status 200, so check that we got a real zip.
        if ! unzip -tq $LLVM_ZIP_FILENAME > /dev/null ; then
            echo "ERROR: LLVM download from google drive is not a valid zip file."
            echo "Google Drive may have exceeded its download quota. Contents:"
            head -c 2000 $LLVM_ZIP_FILENAME
            echo
            exit 1
        fi
        unzip $LLVM_ZIP_FILENAME > /dev/null
        rm -f $LLVM_ZIP_FILENAME
    fi
    export LLVM_ROOT=$PWD/llvm-build
    popd
else
    export LLVM_ROOT=$LLVM_PATH
fi
echo "LLVM_ROOT = $LLVM_ROOT"


mkdir -p winflexbison
pushd winflexbison
WFBZIP=win_flex_bison-2.5.25.zip
curl $CURL_RETRY_FLAGS --location https://github.com/lexxmark/winflexbison/releases/download/v2.5.25/$WFBZIP -o $WFBZIP
unzip $WFBZIP
export FLEX_ROOT=$PWD
export BISON_ROOT=$PWD
ls .
popd


# Save the env for use by other stages
src/build-scripts/save-env.bash
