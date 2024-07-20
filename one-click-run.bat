echo "Creating virtual environment"
python -m venv .okx_venv
echo "Activating virtual environment"
call .okx_venv\Scripts\activate.bat
echo "Installing dependencies"
pip3 install -r requirements.txt
echo "dependencies installed"
echo "virtual environment created and activated"
echo "Running the application"
python main.py