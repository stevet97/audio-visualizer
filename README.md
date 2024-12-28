# 🎶 Real-Time Audio-Reactive AI Visual Generator

## Overview

This project leverages **StyleGAN2-ADA** combined with real-time audio analysis to create **stunning, audio-reactive visuals** for DJs, clubs, and live events. Using advanced machine learning techniques and **TouchDesigner** as the front end, the system generates high-quality visuals synchronized to music, offering DJs real-time control over creative parameters like temperature, truncation, and style diversity.

---

## Key Features

- **Audio-Reactive Visuals**:
  - Visuals adapt to the music's BPM, generating new frames on key beats (e.g., every 4th or 8th beat).
  - Smooth interpolation between frames for continuous and seamless visuals.
  
- **StyleGAN2-ADA**:
  - Generates high-quality, intricate visuals in real time or near real-time.
  - Optimized for efficient frame generation and audio synchronization.

- **Customizable DJ Dashboard**:
  - Real-time controls for parameters like temperature, truncation, top-k, and style selection.
  - BPM synchronization ensures visuals align perfectly with the rhythm.

- **Cloud-Based Processing**:
  - Uses scalable **AWS EC2 GPU instances** to handle computationally intensive tasks.
  - Offers a subscription-based model for DJs, reducing upfront hardware costs.

- **TouchDesigner Integration**:
  - Acts as the real-time rendering engine for live performances.
  - Adds dynamic effects, transitions, and audio-reactive overlays to StyleGAN outputs.

---

## Architecture

1. **Audio Processing**:
   - **Librosa** extracts tempo, beats, and spectral features from the audio in real time.
   - Syncs frame generation with the detected BPM for rhythmic visual updates.

2. **Generative AI**:
   - **VAE (Variational Autoencoder)** processes audio features into latent vectors.
   - **StyleGAN2-ADA** generates visuals based on these latent vectors, with parameters customizable by the user.

3. **TouchDesigner**:
   - Receives generated visuals and applies additional effects (e.g., particle systems, 3D transformations).
   - Acts as the display engine for live events.

4. **Cloud Integration**:
   - **AWS EC2 with NVIDIA A100 GPUs** handles visual generation.
   - **Amazon S3** stores pre-generated visuals for quick retrieval.
   - **AWS IVS** or WebRTC streams visuals in real time to DJs and their audiences.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name

