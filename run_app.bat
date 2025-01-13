@echo off
echo "Launching Streamlit app with conda environment..."

REM (1) Activate the conda environment:
REM     Be sure 'stylegan_ada_env' is the same name you used in 'conda create -n stylegan_ada_env ...'
call conda activate stylegan_ada_env

REM (2) (Optional) install requirements each time. 
REM     If you prefer, remove these lines once the environment is set up permanently.
pip install --upgrade pip
pip install -r requirements.txt

REM (3) Run the Streamlit app
streamlit run app.py

REM (4) Keep the window open so you can see any console output or errors
pause
