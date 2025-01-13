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
# 1) Must be the FIRST Streamlit command!
# --------------------------------------------------
st.set_page_config(page_title="Audio Visualizer + VAE + OSC", layout="wide")

# --------------------------------------------------
# 2) Ensure "is_playing" is initialized
# --------------------------------------------------
if "is_playing" not in st.session_state:
    st.session_state["is_playing"] = False

# --------------------------------------------------
# Simple CSS for a nicer look (optional)
# --------------------------------------------------
st.markdown(
    """
    <style>
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
# (Optional) CUDA / Device Info
# --------------------------------------------------
def check_cuda():
    cuda_available = torch.cuda.is_available()
    st.write("CUDA available:", cuda_available)
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        st.write("Device name:", device_name)

# --------------------------------------------------
# StyleGAN2-ADA model check (example)
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
# 3) Load & Process Audio (Mel-Spectrogram) with colorbar fix
# --------------------------------------------------
def load_and_process_audio(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    # Grab the "mappable" from specshow
    img = librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        fmax=8000,
        ax=ax
    )

    # colorbar references the returned "img" object
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-Spectrogram')
    plt.tight_layout()

    return mel_spec_db, sr, fig

# --------------------------------------------------
# 4) VAE
# --------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)

# --------------------------------------------------
# 5) Run VAE on Mel-Spectrogram
# --------------------------------------------------
def run_vae_on_melspec(mel_spec_db: np.ndarray):
    st.write("Running VAE on Mel-spectrogram...")

    mel_spec_flat = mel_spec_db.flatten().astype(np.float32)
    input_dim = mel_spec_flat.size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_dim=input_dim, hidden_dim=1024, latent_dim=512).to(device)

    mel_tensor = torch.tensor(mel_spec_flat).unsqueeze(0).to(device)

    latent = vae(mel_tensor)  # shape: [1, 512]
    latent_vector = latent.detach().cpu().numpy().flatten()  # shape: (512,)

    # Map from [-1,1] to [0,1]
    latent_vector_mapped = (latent_vector + 1.0) / 2.0
    latent_vector_mapped = np.clip(latent_vector_mapped, 0.0, 1.0)

    st.write("VAE processed successfully. Final latent shape:", latent_vector_mapped.shape)
    return latent_vector_mapped


# --------------------------------------------------
# 6) Send several transitions
# --------------------------------------------------
def send_multiple_transitions(ip, port, steps, sleep_time, min_range, max_range, num_transitions):
    """
    Sends 'num_transitions' consecutive transitions, each from current_vec
    to a new random vector, blocking until they're all finished.
    """
    client = udp_client.SimpleUDPClient(ip, port)
    st.write(f"Sending {num_transitions} transitions to TouchDesigner at {ip}:{port} ...")

    current_vec = np.random.uniform(min_range, max_range, 3)

    for t in range(num_transitions):
        new_vec = np.random.uniform(min_range, max_range, 3)
        st.write(f"Transition {t+1}/{num_transitions}: {current_vec} -> {new_vec}")

        for i in range(steps):
            frac = i / float(steps - 1) if steps > 1 else 1.0
            interpolated_vec = current_vec + (new_vec - current_vec) * frac
            r, g, b = interpolated_vec

            client.send_message("/latent_vector/r", float(r))
            client.send_message("/latent_vector/g", float(g))
            client.send_message("/latent_vector/b", float(b))

            time.sleep(sleep_time)

        current_vec = new_vec

    st.success("All transitions complete!")

# --------------------------------------------------
# 7) Streamlit UI
# --------------------------------------------------
def main():
    st.title("Audio Visualizer + VAE + OSC (Enhanced)")

    st.write("""
    This interface demonstrates an audio-driven VAE pipeline, plus OSC to TouchDesigner.
    Check CUDA for GPU support, load & process audio for a mel-spectrogram, 
    run a VAE, and send random vectors to TouchDesigner.
    """)

    # (Optional) Check CUDA
    if st.checkbox("Check CUDA Info"):
        check_cuda()

    # Example paths
    model_path_default = r"C:\AI\stylegan2-ada-main\ffhq.pkl"
    audio_file_default = r"C:\Users\steph\OneDrive\Desktop\Audio Visualizer\Sample audio\EQed Audio.wav"

    st.subheader("StyleGAN Model Setup")
    model_path = st.text_input("StyleGAN-ADA model path:", value=model_path_default)
    if st.button("Ensure/Download StyleGAN model"):
        ensure_stylegan_weights(model_path)

    # Audio Section
    st.subheader("Audio File")
    audio_path = st.text_input("Path to Audio:", value=audio_file_default)
    if st.button("Load & Process Audio on Mel-Spectrogram"):
        mel_spec_db, sr, fig = load_and_process_audio(audio_path)
        st.session_state["mel_spec_db"] = mel_spec_db
        # Show the matplotlib figure in Streamlit
        # create 3 columns: left, center, right
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.pyplot(fig)
            st.caption("Mel-spectrogram: a time-frequency representation (dB scale) of your audio data.")

    # OSC Controls
    st.subheader("TouchDesigner OSC Controls")
    ip = st.text_input("TouchDesigner IP", "127.0.0.1", help="IP address for TouchDesigner")
    port = st.number_input("TouchDesigner Port", value=7000)
    steps = st.slider("Number of Steps", 1, 200, 50, help="Higher = smoother transitions")
    sleep_time = st.slider("Sleep Time (seconds)", 0.0001, 1.0, 0.02, step=0.0001)
    min_range = st.number_input("Min Random Range", value=-2.0)
    max_range = st.number_input("Max Random Range", value=2.0)

    st.write("---")
    st.markdown("### Multiple Transitions (No Threading)")

    # Let user choose either 20, 50, or a custom 51–200 transitions
    trans_choice = st.radio(
        "How many transitions do you want to send?",
        options=["20 transitions", "50 transitions", "Custom"]
    )

    # We'll store the final number of transitions in num_transitions
    num_transitions = 0
    if trans_choice == "20 transitions":
        num_transitions = 20
    elif trans_choice == "50 transitions":
        num_transitions = 50
    else:
        # Show a slider for custom transitions in the range 51..200
        num_transitions = st.slider("Custom Transitions (51–200)", 51, 200, 100)

    if st.button("Send Multiple Transitions"):
        send_multiple_transitions(
            ip=ip,
            port=port,
            steps=steps,
            sleep_time=sleep_time,
            min_range=min_range,
            max_range=max_range,
            num_transitions=num_transitions
        )

    st.write("---")
    st.write("This 'Multiple Transitions' approach sends consecutive transitions back-to-back, "
             "blocking the UI until it completes. For an indefinite or stoppable stream, you'd "
             "re-introduce the threading approach.")

if __name__ == "__main__":
    main()
