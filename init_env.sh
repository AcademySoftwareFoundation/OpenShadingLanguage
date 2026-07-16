#!/bin/bash

echo ">>> 1. Odblokowanie systemu i czyszczenie błędów (Opcja Nuklearna)..."
# Usuwamy problematyczne skrypty postinst, które blokowały dpkg brakiem py3compile
sudo rm -f /var/lib/dpkg/info/python3-yaml.postinst
sudo rm -f /var/lib/dpkg/info/python3-pygments.postinst
sudo rm -f /var/lib/dpkg/info/libboost-mpi-python1.83.0.postinst

# Wymuszamy naprawę systemu i usunięcie zepsutych pakietów
sudo dpkg --configure -a
sudo apt-get install -f -y

echo ">>> 2. Instalacja bezpiecznych zależności (minimalny zestaw)..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    clang \
    llvm-dev \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-system-dev \
    libopenexr-dev \
    libopenimageio-dev \
    libpugixml-dev \
    libtbb-dev \
    zlib1g-dev \
    python3-minimal

echo ">>> 3. Konfiguracja AMD ROCm..."
if [ ! -f /etc/apt/keyrings/rocm.gpg ]; then
    sudo mkdir -p /etc/apt/keyrings
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
fi

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/latest/ noble main" | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt-get update
sudo apt-get install -y rocm-hip-sdk hipcc

echo ">>> 4. Eksportowanie zmiennych środowiskowych..."
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export PATH=$PATH:$ROCM_PATH/bin:$HIP_PATH/bin

sudo apt-get update
sudo apt-get install -y pybind11-dev
sudo apt-get install -y bison flex
sudo apt-get install -y libclang-18-dev libclang-dev
echo "--- Środowisko gotowe do kompilacji ---"
