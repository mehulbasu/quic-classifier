#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

git config --global user.name "Mehul Basu"
git config --global user.email "mehulbasu@gmail.com"

echo "Updating package list..."
sudo apt update -y

# echo "Installing pip..."
# sudo apt install -y python3-pip

# echo "Installing Python 3.10 venv package..."
# sudo apt install -y python3.10-venv

# Downloading Conda installer
echo "Downloading Miniforge installer..."
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge3-Linux-x86_64.sh

# Installing Conda
echo "Installing Miniforge..."
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
export PATH="$HOME/miniforge3/bin:$PATH"

# Remove installer
rm Miniforge3-Linux-x86_64.sh

# Install CUDA
echo "Downloading CUDA keyring package..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get -y install cuda-toolkit-12-8
rm cuda-keyring_1.1-1_all.deb

# Set up environment variables for CUDA
echo "Setting up CUDA environment variables..."
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

echo "Installing Ubuntu drivers common package..."
sudo apt install -y ubuntu-drivers-common

Run this command to list your GPU and the recommended driver
ubuntu-drivers devices

# echo "Installing NVIDIA driver 580..."
# sudo apt install -y nvidia-driver-580

# echo "All installations complete. Rebooting system..."
# sudo reboot
