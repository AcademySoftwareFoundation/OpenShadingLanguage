#!/usr/bin/env bash
#

set -ex

echo "Which g++ " `which g++`
g++ --version && true
ls /usr/bin/g++* && true
/usr/bin/g++ --version && true

export PATH=/opt/rh/devtoolset-6/root/usr/bin:/usr/local/bin:$PATH
#ls /opt/rh/devtoolset-6/root/usr/bin && true
#ls /usr/local/bin

ls /etc/yum.repos.d

#sudo yum install -y giflib giflib-devel && true
sudo yum install -y Field3D Field3D-devel && true

#sudo rpm -v --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro && true
#sudo rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm && true
#sudo yum install -y ffmpeg ffmpeg-devel && true



if [[ "$LLVM_BRANCH" != ""  || "$LLVM_VERSION" != "" ]] ; then
    source src/build-scripts/build_llvm.bash
fi


# Build or download LLVM
#source src/build-scripts/build_llvm.bash

source src/build-scripts/build_pybind11.bash

# Only build OpenEXR if a specific version is requested
if [[ "$OPENEXR_BRANCH" != "" || "$OPENEXR_VERSION" != "" ]] ; then
    source src/build-scripts/build_openexr.bash
fi

# Only build OpenColorIO if a specific version is requested
if [[ "$OCIO_BRANCH" != ""  || "$OCIO_VERSION" != "" ]] ; then
    # Temporary (?) fix: GH ninja having problems, fall back to make
    CMAKE_GENERATOR="Unix Makefiles" \
    source src/build-scripts/build_ocio.bash
fi

# Only build OpenImageIO if a specific version is requested
if [[ "$OPENIMAGEIO_BRANCH" != ""  || "$OPENIMAGEIO_VERSION" != "" ]] ; then
    # There are many parts of OIIO we don't need to build
    export ENABLE_iinfo=0 ENABLE_iv=0 ENABLE_igrep=0 ENABLE_iconvert=0 ENABLE_testtex=0
    export ENABLE_cineon=0 ENABLE_DDS=0 ENABLE_DPX=0 ENABLE_FITS=0
    export ENABLE_iff=0 ENABLE_jpeg2000=0 ENABLE_PNM=0 ENABLE_PSD=0
    export ENABLE_RLA=0 ENABLE_SGI=0 ENABLE_SOCKET=0 ENABLE_SOFTIMAGE=0
    export ENABLE_TARGA=0 ENABLE_WEBP=0
    export OPENIMAGEIO_MAKEFLAGS="OIIO_BUILD_TESTS=0 USE_OPENGL=0"
    source src/build-scripts/build_openimageio.bash
fi
