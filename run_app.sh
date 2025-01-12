#!/bin/bash
# ============================================
# Shell script for macOS or Linux users
# ============================================

echo "Launching Streamlit app on macOS/Linux..."

# --- If using conda, activate the environment (change the name if needed)
conda activate
# If 'conda init' properly set up, this should work.
# If not, might need 'source /opt/miniconda3/etc/profile.d/conda.sh' first
conda activate stylegan_ada_env

# --- Run the Streamlit app
streamlit run app.py

# Optional: keep the terminal open
# read -p "Press any key to close..."
