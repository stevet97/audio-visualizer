@echo off
echo "Launching Streamlit app with local venv..."

REM (1) Create a local venv if none exists:
if not exist .venv (
    python -m venv .venv
)

REM (2) Activate the venv:
call .venv\Scripts\activate

REM (3) Install requirements:
pip install --upgrade pip
pip install -r requirements.txt

REM (4) Run the app
python -m streamlit run app.py

pause
