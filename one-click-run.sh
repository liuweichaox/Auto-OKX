#!/bin/bash

echo "Creating virtual environment"
python3 -m venv .okx_venv
echo "Activating virtual environment"
source .okx_venv/bin/activate
echo "Installing dependencies"
pip3 install  -r requirements.txt
echo "Dependencies installed"
echo "Virtual environment created and activated"
echo "Running the application"
python3 main.py