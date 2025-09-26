#!/bin/bash

# --- Colors ---
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# --- Environment check ---
if [ -z "$CONDA_PREFIX" ]; then
    echo -e "${RED}Please activate your Conda environment first.${NC}"
    return 0 2>/dev/null || exit 0
fi

# --- Check if the script is sourced ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "${YELLOW}Please run this script as:"
    echo -e "    ${NC}source $(basename "${BASH_SOURCE[0]}")"
    echo -e "Otherwise, '${YELLOW}conda activate${NC}' won't persist in your current shell."
    return 0 2>/dev/null || exit 0
fi

# --- Environment variables ---
export ZLIB_ROOT=$CONDA_PREFIX
export LLVM_ROOT=$CONDA_PREFIX
export LLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
export CMAKE_PREFIX_PATH=$LLVM_DIR:$CMAKE_PREFIX_PATH
export CC=$CONDA_PREFIX/bin/clang
export CXX=$CONDA_PREFIX/bin/clang++

# --- Project directories ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSL_DIR="$(realpath "$SCRIPT_DIR/../../")"
PROJECT_BUILD_DIR="$(realpath "$OSL_DIR/../osl-build")"
PROJECT_INSTALL_DIR="$(realpath "$OSL_DIR/../osl-install")"
BUILD_DIR_RELEASE="$PROJECT_BUILD_DIR/release"
BUILD_DIR_DEBUG="$PROJECT_BUILD_DIR/debug"

# --- Out-of-source build notice ---
NOTICE_FILE="$PROJECT_BUILD_DIR/.notice_shown"
if [ ! -f "$NOTICE_FILE" ]; then
    echo -e "${YELLOW}Notice:${NC} This script will create out-of-source build directories:"
    echo -e "  Debug build:   $BUILD_DIR_DEBUG"
    echo -e "  Release build: $BUILD_DIR_RELEASE"
    echo -e "  Install dir:   $PROJECT_INSTALL_DIR"

    read -p "Do you want to continue? [y/n]: " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Build cancelled by user."
        return 0 2>/dev/null || exit 0
    fi

    mkdir -p "$PROJECT_BUILD_DIR"
    touch "$NOTICE_FILE"
fi

# --- Load or ask for user configuration ---
CONFIG_FILE="$PROJECT_BUILD_DIR/.osl_build_config"
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}Loading previous build configuration...${NC}"
    source "$CONFIG_FILE"
else
    echo -e "${YELLOW}No configuration found. Please select build options:${NC}"

    # Boolean options
    read -p "Build tests? [ON/OFF] " OSL_BUILD_TESTS
    read -p "Build shaders? [ON/OFF] " OSL_BUILD_SHADERS
    read -p "Include OptiX support? [ON/OFF] " OSL_USE_OPTIX
    read -p "Use fast math? [ON/OFF] " USE_FAST_MATH
    read -p "Qt support? [ON/OFF] " USE_QT
    read -p "Use LLVM bitcode? [ON/OFF] " USE_LLVM_BITCODE
    read -p "Build plugins? [ON/OFF] " OSL_BUILD_PLUGINS
    read -p "Use ustring hash? [ON/OFF] " OSL_USTRINGREP_IS_HASH
    read -p "No default TextureSystem? [ON/OFF] " OSL_NO_DEFAULT_TEXTURESYSTEM
    read -p "OIIO SIMD-friendly fmath? [ON/OFF] " OIIO_FMATH_SIMD_FRIENDLY
    read -p "Install documentation? [ON/OFF] " INSTALL_DOCS
    read -p "Include patch in inner namespace? [ON/OFF] " PROJ_INNER_NAMESPACE_INCLUDE_PATCH

    # Strings / paths
    read -p "Outer namespace (leave empty for default 'OSL'): " PROJ_OUTER_NAMESPACE
    read -p "Debug postfix (leave empty for none): " CMAKE_DEBUG_POSTFIX
    read -p "OSL lib name suffix (leave empty for none): " OSL_LIBNAME_SUFFIX
    read -p "CUDA target architecture (e.g., sm_60): " CUDA_TARGET_ARCH
    read -p "Extra CUDA libs (space-separated): " CUDA_EXTRA_LIBS
    read -p "Extra OptiX libs (space-separated): " OPTIX_EXTRA_LIBS

    # Save choices
    mkdir -p "$PROJECT_BUILD_DIR"
    cat > "$CONFIG_FILE" <<EOL
OSL_BUILD_TESTS=${OSL_BUILD_TESTS:-ON}
OSL_BUILD_SHADERS=${OSL_BUILD_SHADERS:-ON}
OSL_USE_OPTIX=${OSL_USE_OPTIX:-OFF}
USE_FAST_MATH=${USE_FAST_MATH:-ON}
USE_QT=${USE_QT:-OFF}
USE_LLVM_BITCODE=${USE_LLVM_BITCODE:-ON}
OSL_BUILD_PLUGINS=${OSL_BUILD_PLUGINS:-ON}
OSL_USTRINGREP_IS_HASH=${OSL_USTRINGREP_IS_HASH:-OFF}
OSL_NO_DEFAULT_TEXTURESYSTEM=${OSL_NO_DEFAULT_TEXTURESYSTEM:-OFF}
OIIO_FMATH_SIMD_FRIENDLY=${OIIO_FMATH_SIMD_FRIENDLY:-OFF}
INSTALL_DOCS=${INSTALL_DOCS:-ON}
PROJ_INNER_NAMESPACE_INCLUDE_PATCH=${PROJ_INNER_NAMESPACE_INCLUDE_PATCH:-ON}
PROJ_OUTER_NAMESPACE="${PROJ_OUTER_NAMESPACE:-OSL}"
CMAKE_DEBUG_POSTFIX="${CMAKE_DEBUG_POSTFIX:-}"
OSL_LIBNAME_SUFFIX="${OSL_LIBNAME_SUFFIX:-}"
CUDA_TARGET_ARCH="${CUDA_TARGET_ARCH:-sm_60}"
CUDA_EXTRA_LIBS="${CUDA_EXTRA_LIBS:-}"
OPTIX_EXTRA_LIBS="${OPTIX_EXTRA_LIBS:-}"
EOL
fi

# --- Function to configure cmake ---
configure_cmake() {
    local BUILD_DIR="$1"
    local BUILD_TYPE="$2"

    mkdir -p "$BUILD_DIR"

    cmake -B "$BUILD_DIR" -S "$OSL_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$PROJECT_INSTALL_DIR" \
        -DSTOP_ON_WARNING=0 \
        -DUSE_QT="$USE_QT" \
        -DQt5_DIR=IGNORE \
        -DQt6_DIR=IGNORE \
        -DOSL_BUILD_TESTS="$OSL_BUILD_TESTS" \
        -DOSL_BUILD_SHADERS="$OSL_BUILD_SHADERS" \
        -DOSL_USE_OPTIX="$OSL_USE_OPTIX" \
        -DUSE_FAST_MATH="$USE_FAST_MATH" \
        -DUSE_LLVM_BITCODE="$USE_LLVM_BITCODE" \
        -DOSL_BUILD_PLUGINS="$OSL_BUILD_PLUGINS" \
        -DOSL_USTRINGREP_IS_HASH="$OSL_USTRINGREP_IS_HASH" \
        -DOSL_NO_DEFAULT_TEXTURESYSTEM="$OSL_NO_DEFAULT_TEXTURESYSTEM" \
        -DOIIO_FMATH_SIMD_FRIENDLY="$OIIO_FMATH_SIMD_FRIENDLY" \
        -DINSTALL_DOCS="$INSTALL_DOCS" \
        -DOSL_OUTER_NAMESPACE="$PROJ_OUTER_NAMESPACE" \
        -DOSL_INNER_NAMESPACE_INCLUDE_PATCH="$PROJ_INNER_NAMESPACE_INCLUDE_PATCH" \
        -DCMAKE_DEBUG_POSTFIX="$CMAKE_DEBUG_POSTFIX" \
        -DOSL_LIBNAME_SUFFIX="$OSL_LIBNAME_SUFFIX" \
        -DCUDA_TARGET_ARCH="$CUDA_TARGET_ARCH" \
        -DCUDA_EXTRA_LIBS="$CUDA_EXTRA_LIBS" \
        -DOPTIX_EXTRA_LIBS="$OPTIX_EXTRA_LIBS"
}

# --- Debug build ---
build_debug() {
    echo -e "${GREEN}=== Building OSL (Debug) ===${NC}"
    configure_cmake "$BUILD_DIR_DEBUG" "Debug"
    cmake --build "$BUILD_DIR_DEBUG" -j$(nproc)
    echo -e "${GREEN}=== Debug build complete ===${NC}"
}

# --- Release build ---
build_release() {
    echo -e "${GREEN}=== Building OSL (Release) ===${NC}"
    mkdir -p "$PROJECT_INSTALL_DIR"
    configure_cmake "$BUILD_DIR_RELEASE" "Release"
    cmake --build "$BUILD_DIR_RELEASE" -j$(nproc)
    echo -e "${GREEN}=== Release build complete ===${NC}"
}

# --- Install from Release build ---
install_release() {
    echo -e "${RED}=== Installing OSL (Release) ===${NC}"
    if [ ! -d "$BUILD_DIR_RELEASE" ]; then
        echo "Error: Release build not found. Run 'source build.sh release' first."
        return 1 2>/dev/null || exit 0
    fi
    cmake --install "$BUILD_DIR_RELEASE"
    echo -e "${GREEN}=== Installed to $PROJECT_INSTALL_DIR ===${NC}"
}

# --- Parse mode ---
if [ -z "$1" ]; then
    echo -e "${RED}Error: You must specify a mode.${NC}"
    echo "Usage: $0 <mode>"
    echo "Modes: debug | release | install"
    return 1 2>/dev/null || exit 0
fi

MODE="$1"
case "$MODE" in
    debug)
        build_debug
        ;;
    release)
        build_release
        ;;
    install)
        install_release
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        echo "Valid modes: debug | release | install"
        return 1 2>/dev/null || exit 0
        ;;
esac
