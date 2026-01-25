#!/bin/bash

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# --- Colors ---
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color


# --- Check if the script is sourced ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "${YELLOW}Please run this script as:"
    echo -e "    ${NC}source $(basename "${BASH_SOURCE[0]}")"
    echo -e "Otherwise, '${YELLOW}conda activate${NC}' won't persist in your current shell."
    return 0 2>/dev/null || exit 0
fi

# --- Check if Miniconda exists ---
if [ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo -e "${RED}Miniconda not found at ~/miniconda3.${NC}"
    echo -e "Please install Miniconda before running this script"
    echo -e "${YELLOW}Important:${NC} During installation, decline any PATH modifications."
    echo -e "This script activates the environment per session using 'source'."
    return 0 2>/dev/null || exit 0
else
    # Load Conda
    source ~/miniconda3/etc/profile.d/conda.sh

    # Create OSL environment if it doesn't exist
    if ! conda info --envs | grep -q "osl-env"; then
        echo "Creating osl-env Conda environment..."
        conda create -y -n osl-env
        
        # Activate the environment
        conda activate osl-env

        # Install dependencies
        echo "Installing dependencies in osl-env..."
        conda install -y -c conda-forge \
            cmake \
            llvmdev=20.1.8 clangdev clangxx_linux-64 libcxx \
            python=3.12 numpy pybind11 \
            openimageio=2.5 imath flex bison pugixml zlib
    else
        echo -e "${GREEN}osl-env environment already exists.${NC}"
        
        # Activate the environment
        conda activate osl-env
    fi

fi
