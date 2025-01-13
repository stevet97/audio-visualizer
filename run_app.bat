@echo off
echo "Launching Streamlit app with conda environment..."

REM (1) Activate the conda environment:
call conda activate stylegan_ada_env

REM (2) (Optional) install requirements each time. 
pip install --upgrade pip
pip install -r requirements.txt

REM (3) Run the Streamlit app
streamlit run app.py
