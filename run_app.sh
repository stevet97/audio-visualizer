#!/bin/bash
# ============================================
# Shell script to launch Streamlit with conda
# ============================================

echo "Launching Streamlit app with conda environment..."

# (1) Activate the conda environment
# Adjust 'stylegan_ada_env' to match your environment name
# If you haven't run 'conda init' for bash, you may need:
#   source ~/anaconda3/etc/profile.d/conda.sh
conda activate stylegan_ada_env

# (2) (Optional) install requirements each time
# Remove these lines if you prefer manual installation
pip install --upgrade pip
pip install -r requirements.txt

# (3) Run the Streamlit app
streamlit run app.py

# (4) On macOS/Linux, the script ends here.
# If you want to force a pause, you can add:
# read -n 1 -s -r -p "Press any key to close..."
