#!/usr/bin/env bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


set -ex


#
# Install system packages when those are acceptable for dependencies.
#
if [[ "$ASWF_ORG" != ""  ]] ; then
    # Using ASWF CentOS container

    export PATH=/opt/rh/devtoolset-6/root/usr/bin:/usr/local/bin:$PATH

    #ls /etc/yum.repos.d

    sudo /usr/bin/yum install -y giflib giflib-devel && true
    # sudo /usr/bin/yum install -y ffmpeg ffmpeg-devel && true

    if [[ "${CONAN_LLVM_VERSION}" != "" ]] ; then
        mkdir conan
        pushd conan
        # Simple way to conan install just one package:
        #   conan install clang/${CONAN_LLVM_VERSION}@aswftesting/ci_common1 -g deploy -g virtualenv
        # But the below method can accommodate multiple requirements:
        echo "[imports]" >> conanfile.txt
        echo "., * -> ." >> conanfile.txt
        echo "[requires]" >> conanfile.txt
        echo "clang/${CONAN_LLVM_VERSION}@aswftesting/ci_common1" >> conanfile.txt
        time conan install .
        echo "--ls--"
        ls -R .
        export PATH=$PWD/bin:$PATH
        export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
        export LLVM_ROOT=$PWD
        popd
    fi

    if [[ "$CXX" == "icpc" || "$CC" == "icc" || "$USE_ICC" != "" ]] ; then
        # Lock down icc to 2022.1 because newer versions hosted on the Intel
        # repo require a glibc too new for the ASWF CentOS7-based containers
        # we run CI on.
        sudo cp src/build-scripts/oneAPI.repo /etc/yum.repos.d
        sudo /usr/bin/yum install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic-2022.1.0.x86_64
        set +e; source /opt/intel/oneapi/setvars.sh --config oneapi_2022.1.0.cfg; set -e
    elif [[ "$CXX" == "icpc" || "$CC" == "icc" || "$USE_ICC" != "" || "$CXX" == "icpx" || "$CC" == "icx" || "$USE_ICX" != "" ]] ; then
        # Lock down icx to 2023.1 because newer versions hosted on the Intel
        # repo require a libstd++ too new for the ASWF containers we run CI on
        # because their default install of gcc 9 based toolchain.
        sudo cp src/build-scripts/oneAPI.repo /etc/yum.repos.d
        sudo yum install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic-2023.1.0.x86_64
        # sudo yum install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic
        set +e; source /opt/intel/oneapi/setvars.sh; set -e
        echo "Verifying installation of Intel(r) oneAPI DPC++/C++ Compiler:"
        icpx --version
    fi

else
    # Using native Ubuntu runner

    # sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    time sudo apt-get update

    time sudo apt-get -q install -y \
        git cmake ninja-build ccache g++ \
        libboost-dev libboost-thread-dev libboost-filesystem-dev \
        libilmbase-dev libopenexr-dev \
        libtiff-dev libgif-dev libpng-dev \
        flex bison libbison-dev \
        libpugixml-dev \
        libopencolorio-dev

    if [[ "${QT_VERSION:-5}" == "5" ]] ; then
        time sudo apt-get -q install -y \
            qt5-default || /bin/true
    elif [[ "${QT_VERSION}" == "6" ]] ; then
        time sudo apt-get -q install -y \
            qt6-base-dev || /bin/true
    fi

    export CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu:$CMAKE_PREFIX_PATH

    # Nonstandard python versions
    if [[ "${PYTHON_VERSION}" == "3.9" ]] ; then
        time sudo apt-get -q install -y python3.9-dev python3-numpy
        pip3 --version
        pip3 install numpy
    elif [[ "$PYTHON_VERSION" == "2.7" ]] ; then
        time sudo apt-get -q install -y python-dev python-numpy
    else
        pip3 install numpy
    fi

    if [[ "$CXX" == "g++-6" ]] ; then
        time sudo apt-get install -y g++-6
    elif [[ "$CXX" == "g++-7" ]] ; then
        time sudo apt-get install -y g++-7
    elif [[ "$CXX" == "g++-8" ]] ; then
        time sudo apt-get install -y g++-8
    elif [[ "$CXX" == "g++-9" ]] ; then
        time sudo apt-get install -y g++-9
    elif [[ "$CXX" == "g++-10" ]] ; then
        time sudo apt-get install -y g++-10
    elif [[ "$CXX" == "g++-11" ]] ; then
        time sudo apt-get install -y g++-11
    elif [[ "$CXX" == "g++-12" ]] ; then
        time sudo apt-get install -y g++-12
    fi

    if [[ "$CXX" == "icpc" || "$CC" == "icc" || "$USE_ICC" != "" || "$CXX" == "icpx" || "$CC" == "icx" || "$USE_ICX" != "" ]] ; then
        wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
        sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
        echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
        time sudo apt-get update
        time sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic=2022.1.0-3768
        # Because multiple (possibly newer) versions of oneAPI may be installed,
        # use a config file to specify compiler and tbb versions
        # NOTE: oneAPI components have independent version numbering.
        set +e; source /opt/intel/oneapi/setvars.sh --config oneapi_2022.1.0.cfg; set -e
    fi

    source src/build-scripts/build_llvm.bash
fi

if [[ "$CMAKE_VERSION" != "" ]] ; then
    source src/build-scripts/build_cmake.bash
fi
cmake --version


if [[ "$OPTIX_VERSION" != "" ]] ; then
    echo "Requested OPTIX_VERSION = '${OPTIX_VERSION}'"
    mkdir -p $LOCAL_DEPS_DIR/dist/include/internal
    OPTIXLOC=https://developer.download.nvidia.com/redist/optix/v${OPTIX_VERSION}
    for f in optix.h optix_device.h optix_function_table.h \
             optix_function_table_definition.h optix_host.h \
             optix_stack_size.h optix_stubs.h optix_types.h optix_7_device.h \
             optix_7_host.h optix_7_types.h \
             internal/optix_7_device_impl.h \
             internal/optix_7_device_impl_exception.h \
             internal/optix_7_device_impl_transformations.h
        do
        curl --retry 100 -m 120 --connect-timeout 30 \
            $OPTIXLOC/include/$f > $LOCAL_DEPS_DIR/dist/include/$f
    done
    export OptiX_ROOT=$LOCAL_DEPS_DIR/dist
fi



#
# Packages we need to build from scratch.
#

source src/build-scripts/build_pybind11.bash

if [[ "$OPENEXR_VERSION" != "" ]] ; then
    source src/build-scripts/build_openexr.bash
fi

# if [[ "$PUGIXML_VERSION" != "" ]] ; then
    source src/build-scripts/build_pugixml.bash
    export MY_CMAKE_FLAGS+=" -DUSE_EXTERNAL_PUGIXML=1 "
# fi

if [[ "$OPENCOLORIO_VERSION" != "" ]] ; then
    source src/build-scripts/build_opencolorio.bash
fi

if [[ "$OPENIMAGEIO_VERSION" != "" ]] ; then
    # There are many parts of OIIO we don't need to build for OSL's tests
    export ENABLE_iinfo=0 ENABLE_iv=0 ENABLE_igrep=0
    export ENABLE_iconvert=0 ENABLE_testtex=0
    # For speed of compiling OIIO, disable the file formats that we don't
    # need for OSL's tests
    export ENABLE_BMP=0 ENABLE_cineon=0 ENABLE_DDS=0 ENABLE_DPX=0 ENABLE_FITS=0
    export ENABLE_ICO=0 ENABLE_iff=0 ENABLE_jpeg2000=0 ENABLE_PNM=0 ENABLE_PSD=0
    export ENABLE_RLA=0 ENABLE_SGI=0 ENABLE_SOCKET=0 ENABLE_SOFTIMAGE=0
    export ENABLE_TARGA=0 ENABLE_WEBP=0
    # We don't need to run OIIO's tests
    export OPENIMAGEIO_CMAKE_FLAGS+=" -DOIIO_BUILD_TESTING=OFF -DOIIO_BUILD_TESTS=0"
    # Don't let warnings in OIIO break OSL's CI run
    export OPENIMAGEIO_CMAKE_FLAGS+=" -DSTOP_ON_WARNING=OFF"
    export OPENIMAGEIO_CMAKE_FLAGS+=" -DUSE_OPENGL=0"
    export OPENIMAGEIO_CMAKE_FLAGS+=" -DUSE_OPENCV=0 -DUSE_FFMPEG=0 -DUSE_QT=0"
    if [[ "${OPENIMAGEIO_UNITY:-1}" != "0" ]] ; then
        # Speed up the OIIO build by doing a "unity" build. (Note: this is
        # only a savings in CI where there are only 1-2 cores available.)
        export OPENIMAGEIO_CMAKE_FLAGS+=" -DCMAKE_UNITY_BUILD=ON -DCMAKE_UNITY_BUILD_MODE=BATCH"
    fi
    source src/build-scripts/build_openimageio.bash
fi

if [[ "$ABI_CHECK" != "" ]] ; then
    source src/build-scripts/build_abi_tools.bash
fi

# Save the env for use by other stages
src/build-scripts/save-env.bash
