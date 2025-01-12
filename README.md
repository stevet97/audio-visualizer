# Audio Visualizer: Real-Time Latent Vectors from Audio to TouchDesigner

This project demonstrates how to **generate real-time visuals** from audio input using machine learning (VAE) and **stream** latent vectors into **TouchDesigner** for powerful, real-time visual output. The system ties together several key technologies—**PyTorch**, **librosa**, **Streamlit**, and the **Open Sound Control (OSC)** protocol—allowing you to map **audio→ML→visual** in a seamless pipeline. 

## Table of Contents
1. [Overview](#overview)
2. [Key Technologies](#key-technologies)
3. [Project Structure](#project-structure)
4. [Dependencies & Installation](#dependencies--installation)
5. [How to Run (Windows / macOS)](#how-to-run-windows--macos)
6. [TouchDesigner Setup](#touchdesigner-setup)
7. [Under the Hood: ML + Visual Pipeline](#under-the-hood-ml--visual-pipeline)
8. [FAQs / Troubleshooting](#faqs--troubleshooting)

---

## Overview
- **Goal**: Translate **audio** (via mel-spectrogram + Variational Autoencoder) into **latent vectors**, then stream those vectors to **TouchDesigner** for **real-time** generative visuals.
- **User Flow**:  
  1. Double-click a script (`.bat` or `.sh`) to launch a **Streamlit** web app.  
  2. Open a **TouchDesigner** file (`.toe`), which listens for incoming OSC data.  
  3. Adjust parameters in the web UI—like how quickly vectors interpolate—while seeing **immediate** feedback in TouchDesigner.  
- **Highlights**:  
  - Showcases **machine learning** (VAE) for turning audio data into a compressed latent representation.  
  - Streams **real-time** numeric data over **OSC**.  
  - Uses **TouchDesigner** for powerful, GPU-accelerated rendering of the received latent vectors.

---

## Key Technologies
1. **[PyTorch](https://pytorch.org/)**  
   - Used for the **Variational Autoencoder (VAE)**, running on CPU or GPU if available.
2. **[librosa](https://librosa.org/)**  
   - For loading audio and computing **mel-spectrograms**.
3. **[Streamlit](https://streamlit.io/)**  
   - Provides a **web-based UI** to adjust parameters, run the VAE pipeline, and send data in real time.
4. **[python-osc](https://pypi.org/project/python-osc/)**  
   - Sends the latent vectors to **TouchDesigner** using the **OSC** protocol.
5. **[TouchDesigner](https://derivative.ca/)**  
   - Receives the incoming vectors, generating or transforming visuals on the fly.
6. **[Conda / Pip](https://docs.conda.io/)/**  
   - For environment management and Python dependencies.

---

## Project Structure
