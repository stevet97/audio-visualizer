@echo off
REM ============================================
REM Batch script for Windows users
REM ============================================

echo "Launching Streamlit app on Windows..."

REM --- If using conda, activate the environment
REM --- (Change 'audio_viz_env' to your actual env name)
call conda activate stylegan_ada_env

REM --- Run the Streamlit app
streamlit run app.py

REM --- Pause so the window doesn't vanish immediately
pause
