#!/bin/bash

echo "Creating virtual environment"
python3 -m venv .auto_okx_venv
echo "Activating virtual environment"
source .auto_okx_venv/bin/activate
echo "Installing dependencies"
pip3 install  -r requirements.txt
echo "Dependencies installed"
echo "Virtual environment created and activated"
echo "Running the application"
python3 auto-trade.py