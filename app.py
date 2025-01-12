# app.py
import streamlit as st
import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from pythonosc import udp_client
import time
import requests
import threading  # for Play/Stop background streaming

# --------------------------------------------------
# Must be the FIRST Streamlit command!
# --------------------------------------------------
st.set_page_config(
    page_title="Audio Visualizer + VAE + OSC",
    layout="wide",
)

# --------------------------------------------------
# Ensure "is_playing" is initialized
# --------------------------------------------------
if "is_playing" not in st.session_state:
    st.session_state["is_playing"] = False

# --------------------------------------------------
# Simple CSS for a nicer look (optional)
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Example CSS tweaks */
    .css-18e3th9 {
        font-family: "Trebuchet MS", sans-serif;
    }
    .stMarkdown h2, h3, label {
        color: #E66B00 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# 1. Basic CUDA / Device Info (Optional)
# --------------------------------------------------
def check_cuda():
    cuda_available = torch.cuda.is_available()
    st.write("CUDA available:", cuda_available)
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        st.write("Device name:", device_name)

# --------------------------------------------------
# 2. Download/Check StyleGAN2-ADA model
# --------------------------------------------------
def ensure_stylegan_weights(model_path: str):
    if not os.path.exists(model_path):
        st.write("Downloading StyleGAN2-ADA model weights...")
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl'
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.write("Model downloaded successfully.")
    else:
        st.write("Model file already exists:", model_path)

# --------------------------------------------------
# 3. Load & Process Audio (Mel-Spectrogram)
# --------------------------------------------------
def load_and_process_audio(audio_file_path: str):
    st.write("Loading audio file:", audio_file_path)
    y, sr = librosa.load(audio_file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    st.write("Audio file loaded successfully. Mel-spectrogram shape:", mel_spec_db.shape)

    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
    plt.colorbar(format='%+2.0f dB', ax=ax)
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    st.pyplot(fig)  # Show the plot in Streamlit

    return mel_spec_db, sr

# --------------------------------------------------
# 4. Define Variational Autoencoder (VAE)
# --------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

# --------------------------------------------------
# 5. Run VAE on the mel-spectrogram
# --------------------------------------------------
def run_vae_on_melspec(mel_spec_db: np.ndarray):
    st.write("Running VAE on Mel-spectrogram...")

    # Flatten & convert to float32
    mel_spec_flat = mel_spec_db.flatten().astype(np.float32)
    input_dim = mel_spec_flat.size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_dim=input_dim, hidden_dim=1024, latent_dim=512).to(device)

    # Convert to a batch of size 1
    mel_tensor = torch.tensor(mel_spec_flat).unsqueeze(0).to(device)

    # Forward pass
    latent = vae(mel_tensor)  # shape: [1, 512]
    latent_vector = latent.detach().cpu().numpy().flatten()  # shape: (512,)

    # Map from [-1, 1] to [0, 1]
    latent_vector_mapped = (latent_vector + 1.0) / 2.0
    latent_vector_mapped = np.clip(latent_vector_mapped, 0.0, 1.0)

    st.write("VAE processed successfully. Final latent shape:", latent_vector_mapped.shape)
    return latent_vector_mapped

# --------------------------------------------------
# 6A. Single "Transition" of OSC to TouchDesigner
# --------------------------------------------------
def send_single_transition(ip: str, port: int, steps: int, sleep_time: float,
                           min_range: float, max_range: float):
    """
    Sends one random transition from a random current_vec to a random new_vec,
    over 'steps' steps with a short sleep_time between each step.
    """
    client = udp_client.SimpleUDPClient(ip, port)
    st.write(f"Sending data to TouchDesigner at {ip}:{port} ...")

    current_vec = np.random.uniform(-1, 1, 3)
    new_vec = np.random.uniform(min_range, max_range, 3)

    for i in range(steps):
        t = i / float(steps - 1) if steps > 1 else 1.0
        interpolated_vec = current_vec + (new_vec - current_vec) * t
        r, g, b = interpolated_vec

        client.send_message("/latent_vector/r", float(r))
        client.send_message("/latent_vector/g", float(g))
        client.send_message("/latent_vector/b", float(b))
        time.sleep(sleep_time)

    st.success("Transition complete!")

# --------------------------------------------------
# 6B. Continuous streaming with Play/Stop (background thread)
# --------------------------------------------------
def continuous_stream(ip, port, steps, sleep_time, min_range, max_range):
    """
    Continuously sends random vectors to TouchDesigner,
    as long as st.session_state["is_playing"] remains True.
    """
    client = udp_client.SimpleUDPClient(ip, port)
    current_vec = np.random.uniform(-1, 1, 3)

    while st.session_state["is_playing"]:
        new_vec = np.random.uniform(min_range, max_range, 3)

        for i in range(steps):
            # Check if user stopped in the middle
            if not st.session_state["is_playing"]:
                break

            t = i / float(steps - 1) if steps > 1 else 1.0
            interpolated_vec = current_vec + (new_vec - current_vec) * t
            r, g, b = interpolated_vec

            client.send_message("/latent_vector/r", float(r))
            client.send_message("/latent_vector/g", float(g))
            client.send_message("/latent_vector/b", float(b))
            time.sleep(sleep_time)

        current_vec = new_vec

# --------------------------------------------------
# 7. Streamlit UI
# --------------------------------------------------
def main():
    st.title("Audio Visualizer + VAE + OSC (Enhanced)")

    st.write("""
    This interface demonstrates an audio-driven VAE pipeline, plus OSC to TouchDesigner.
    Check CUDA to see if you have GPU support, load & process audio for a mel-spectrogram, 
    run a VAE, and send random vectors to TouchDesigner via OSC.
    """)

    # Optional: Check CUDA
    if st.checkbox("Check CUDA Info"):
        check_cuda()

    # Model paths
    model_path_default = r"C:\AI\stylegan2-ada-main\ffhq.pkl"
    audio_file_default = r"C:\Users\steph\OneDrive\Desktop\Audio Visualizer\Sample audio\EQed Audio.wav"

    st.subheader("StyleGAN Model Setup")
    model_path = st.text_input("StyleGAN-ADA model path:", value=model_path_default)
    if st.button("Ensure/Download StyleGAN model"):
        ensure_stylegan_weights(model_path)

    st.subheader("Audio File")
    audio_path = st.text_input("Path to Audio:", value=audio_file_default)
    if st.button("Load & Process Audio"):
        mel_spec_db, sr = load_and_process_audio(audio_path)
        st.session_state["mel_spec_db"] = mel_spec_db

    # VAE Step
    if "mel_spec_db" in st.session_state:
        if st.button("Run VAE on Mel-Spectrogram"):
            latent_vec = run_vae_on_melspec(st.session_state["mel_spec_db"])
            st.session_state["vae_latent"] = latent_vec

    # Show shape of stored latent if it exists
    if "vae_latent" in st.session_state:
        st.write(f"Current stored latent shape: {st.session_state['vae_latent'].shape}")

    st.subheader("TouchDesigner OSC Controls")
    ip = st.text_input("TouchDesigner IP", "127.0.0.1", help="IP address for TouchDesigner")
    port = st.number_input("TouchDesigner Port", value=7000, help="Port that TD is listening on")
    steps = st.slider("Number of Steps", 1, 200, 50, help="Higher = smoother transitions")
    sleep_time = st.slider("Sleep Time (seconds)", 0.0001, 1.0, 0.02, step=0.0001,
                           help="Delay between each step (smaller = faster updates)")
    min_range = st.number_input("Min Random Range", value=-2.0, help="Lower bound for random vector")
    max_range = st.number_input("Max Random Range", value=2.0, help="Upper bound for random vector")

    # Single transition
    if st.button("Send One Transition"):
        send_single_transition(ip, port, steps, sleep_time, min_range, max_range)

    st.write("---")
    st.markdown("### Continuous Streaming (Play/Stop)")

    # Play/Stop approach for background streaming
    def start_play():
        st.session_state["is_playing"] = True
        threading.Thread(
            target=continuous_stream,
            args=(ip, port, steps, sleep_time, min_range, max_range),
            daemon=True
        ).start()

    def stop_play():
        st.session_state["is_playing"] = False

    colA, colB = st.columns([1,1])
    with colA:
        st.write("Press **Play** to start infinite random vectors, interpolated continuously.")
        if st.button("Play", disabled=st.session_state["is_playing"]):
            start_play()
    with colB:
        st.write("Press **Stop** to end the continuous loop.")
        if st.button("Stop", disabled=not st.session_state["is_playing"]):
            stop_play()

    if st.session_state["is_playing"]:
        st.success("Streaming in progress... (Click Stop to end)")
    else:
        st.info("Not streaming. (Click Play to start continuous random vectors)")

    st.write("---")
    st.write("Note: The single transition code remains for quick one-off tests. The Play/Stop code uses a background thread to avoid freezing the Streamlit UI.")


if __name__ == "__main__":
    main()
