#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Updating package list..."
sudo apt update -y

echo "Installing pip..."
sudo apt install -y python3-pip

echo "Installing Python 3.10 venv package..."
sudo apt install -y python3.10-venv

echo "Installing Ubuntu drivers common package..."
sudo apt install -y ubuntu-drivers-common

echo "Installing CUDA toolkit..."
sudo apt install -y nvidia-cuda-toolkit

echo "Installing NVIDIA driver 580..."
sudo apt install -y nvidia-driver-580

echo "All installations complete. Rebooting system..."
sudo reboot
