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

# Remove installer
rm Miniforge3-Linux-x86_64.sh

echo "Installing Ubuntu drivers common package..."
sudo apt install -y ubuntu-drivers-common

echo "Installing CUDA toolkit..."
sudo apt install -y nvidia-cuda-toolkit

# Run this command to list your GPU and the recommended driver
# ubuntu-drivers devices

# echo "Installing NVIDIA driver 580..."
# sudo apt install -y nvidia-driver-580

echo "All installations complete. Rebooting system..."
sudo reboot
