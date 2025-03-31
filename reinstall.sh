#!/bin/bash

# Stop on any error
set -e

echo "Uninstalling existing stereodemo package..."
pip uninstall -y stereodemo || true

echo "Installing required dependencies..."
pip install Syphon-python

echo "Installing stereodemo package in development mode..."
pip install -e .

echo -e "\nIMPORTANT: Run stereodemo using python -m, NOT as a direct command:"
echo "  python -m stereodemo --syphon              # For normal mode with Syphon"
echo "  python -m stereodemo --calibrate --syphon  # For calibration mode with Syphon"
echo "  python -m stereodemo --calibrate --oak     # For calibration mode with OAK-D camera"
echo -e "\nDO NOT run 'stereodemo' directly as it's not installed as a command-line tool." 