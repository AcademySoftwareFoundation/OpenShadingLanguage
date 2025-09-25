#!/bin/bash
set -e  # exit on any error

# --- Colors ---
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# --- Environment check ---
if [ -z "$CONDA_PREFIX" ]; then
    echo -e "${RED}Please activate your Conda environment first.${NC}"
    exit 1
fi

# --- Common environment variables ---
export ZLIB_ROOT=$CONDA_PREFIX
export LLVM_ROOT=$CONDA_PREFIX
export LLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm
export CMAKE_PREFIX_PATH=$LLVM_DIR:$CMAKE_PREFIX_PATH
export CC=$CONDA_PREFIX/bin/clang
export CXX=$CONDA_PREFIX/bin/clang++

# --- Project directories ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSL_DIR="$SCRIPT_DIR"
PROJECT_BUILD_DIR="$SCRIPT_DIR/../osl-build"
PROJECT_INSTALL_DIR="$SCRIPT_DIR/../osl-install"

BUILD_DIR_RELEASE="$PROJECT_BUILD_DIR/release"
BUILD_DIR_DEBUG="$PROJECT_BUILD_DIR/debug"

# --- Out-of-source build notice (only once) ---
NOTICE_FILE="$PROJECT_BUILD_DIR/.notice_shown"

if [ ! -f "$NOTICE_FILE" ]; then
    echo -e "${YELLOW}Notice:${NC} This script will create out-of-source build directories:"
    echo -e "  Debug build:   $BUILD_DIR_DEBUG"
    echo -e "  Release build: $BUILD_DIR_RELEASE"
    echo -e "  Install dir:   $PROJECT_INSTALL_DIR"

    read -p "Do you want to continue? [y/n]: " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Build cancelled by user."
        exit 0
    fi

    mkdir -p "$PROJECT_BUILD_DIR"
    touch "$NOTICE_FILE"
fi

# --- Debug build ---
build_debug() {
    echo -e "${GREEN}=== Building OSL (Debug) ===${NC}"
    mkdir -p "$BUILD_DIR_DEBUG"

    cmake -B "$BUILD_DIR_DEBUG" -S "$OSL_DIR" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSTOP_ON_WARNING=0 \
        -DUSE_QT=OFF \
        -DQt5_DIR=IGNORE \
        -DQt6_DIR=IGNORE

    cmake --build "$BUILD_DIR_DEBUG" -j$(nproc)
    echo -e "${GREEN}=== Debug build complete ===${NC}"
}

# --- Release build ---
build_release() {
    echo -e "${GREEN}=== Building OSL (Release) ===${NC}"
    mkdir -p "$BUILD_DIR_RELEASE"
    mkdir -p "$PROJECT_INSTALL_DIR"

    cmake -B "$BUILD_DIR_RELEASE" -S "$OSL_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PROJECT_INSTALL_DIR" \
        -DSTOP_ON_WARNING=0 \
        -DUSE_QT=OFF \
        -DQt5_DIR=IGNORE \
        -DQt6_DIR=IGNORE

    cmake --build "$BUILD_DIR_RELEASE" -j$(nproc)
    echo -e "${GREEN}=== Release build complete ===${NC}"
}

# --- Install from Release build ---
install_release() {
    echo -e "${RED}=== Installing OSL (Release) ===${NC}"
    if [ ! -d "$BUILD_DIR_RELEASE" ]; then
        echo "Error: Release build not found. Run './build.sh release' first."
        exit 1
    fi
    cmake --install "$BUILD_DIR_RELEASE"
    echo -e "${GREEN}=== Installed to $PROJECT_INSTALL_DIR ===${NC}"
}

# --- Parse command ---
if [ -z "$1" ]; then
    echo -e "${RED}Error: You must specify a mode.${NC}"
    echo "Usage: $0 <mode>"
    echo "Modes: debug | release | install"
    exit 1
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
        exit 1
        ;;
esac

