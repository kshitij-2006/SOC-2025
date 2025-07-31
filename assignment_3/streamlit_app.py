# morphix_streamlit_app.py

import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pickle

st.set_page_config(layout="wide")

# ------------------------- Constants -------------------------
PRESET_NAMES = ["Glasses", "Beard", "Child", "Cartoon", "Artistic", "Smile", "Age", "Gender"]
PRESET_DIR = Path("presets")  # should contain .npy files for directions
STYLEGAN_PATH = Path("stylegan2-ada-pytorch/pretrained/ffhq.pkl")

# ------------------------- Load Model -------------------------
@st.cache_resource
def load_generator():
    with open(STYLEGAN_PATH, 'rb') as f:
        G = pickle.load(f)['G_ema'].eval()  # Removed .cuda()
    return G

# ------------------------- Utils -------------------------
def generate_image(G, w_plus):
    image = G.synthesis(w_plus, noise_mode='const')
    image = (image.clamp(-1, 1) + 1) * 127.5
    image = image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)

def apply_direction(w, direction, alpha=1.0):
    return w + alpha * direction.to(w.device)

def load_directions():
    directions = {}
    for name in PRESET_NAMES:
        path = PRESET_DIR / f"{name.lower()}.npy"
        if path.exists():
            directions[name] = torch.tensor(np.load(path))  # Removed .cuda()
    return directions

def style_mix(w1, w2, crossover):
    mixed = w1.clone()
    mixed[:, crossover:] = w2[:, crossover:]
    return mixed

# ------------------------- Initialize Session State -------------------------
if 'G' not in st.session_state:
    st.session_state.G = load_generator()
    st.session_state.current_w = None
    st.session_state.display_image = None
    st.session_state.undo_stack = []
    st.session_state.redo_stack = []
    st.session_state.secondary_w = None
    st.session_state.directions = load_directions()

G = st.session_state.G

# ------------------------- Header -------------------------
st.title("🎭 Morphix: GAN Face Editor & Mixer")

# ------------------------- Generate Random Face -------------------------
if st.button("🔀 Generate Random Face"):
    z = torch.randn(1, G.z_dim)
    w = G.mapping(z, None)
    w_plus = w.unsqueeze(1).repeat(1, G.num_ws, 1)
    img = generate_image(G, w_plus)
    st.session_state.current_w = w_plus
    st.session_state.display_image = img
    st.session_state.undo_stack.clear()
    st.session_state.redo_stack.clear()

# ------------------------- Display Current Image -------------------------
if st.session_state.display_image:
    st.image(st.session_state.display_image, caption="Current Face", width=512)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("↩️ Undo") and st.session_state.undo_stack:
            st.session_state.redo_stack.append(st.session_state.current_w)
            st.session_state.current_w = st.session_state.undo_stack.pop()
            st.session_state.display_image = generate_image(G, st.session_state.current_w)

    with col2:
        if st.button("↪️ Redo") and st.session_state.redo_stack:
            st.session_state.undo_stack.append(st.session_state.current_w)
            st.session_state.current_w = st.session_state.redo_stack.pop()
            st.session_state.display_image = generate_image(G, st.session_state.current_w)

    with col3:
        st.download_button("📥 Download Image", data=st.session_state.display_image.tobytes(),
                           file_name="face.png", mime="image/png")

    # ------------------------- Attribute Sliders -------------------------
    st.markdown("### 🎚️ Adjust Attributes")
    col1, col2, col3 = st.columns(3)
    smile = col1.slider("Smile", -5.0, 5.0, 0.0, step=0.1)
    age = col2.slider("Age", -5.0, 5.0, 0.0, step=0.1)
    gender = col3.slider("Gender", -5.0, 5.0, 0.0, step=0.1)

    if st.button("✨ Apply Attribute Changes"):
        st.session_state.undo_stack.append(st.session_state.current_w.clone())
        w = st.session_state.current_w
        if "Smile" in st.session_state.directions:
            w = apply_direction(w, st.session_state.directions["Smile"], smile)
        if "Age" in st.session_state.directions:
            w = apply_direction(w, st.session_state.directions["Age"], age)
        if "Gender" in st.session_state.directions:
            w = apply_direction(w, st.session_state.directions["Gender"], gender)
        st.session_state.current_w = w
        st.session_state.display_image = generate_image(G, w)
        st.session_state.redo_stack.clear()

# ------------------------- Preset Edits -------------------------
st.markdown("### 🎨 Preset Edits")
preset_cols = st.columns(len(PRESET_NAMES))

for i, name in enumerate(PRESET_NAMES):
    if preset_cols[i].button(name):
        direction = st.session_state.directions.get(name)
        if direction is not None and st.session_state.current_w is not None:
            st.session_state.undo_stack.append(st.session_state.current_w.clone())
            new_w = apply_direction(st.session_state.current_w, direction)
            st.session_state.current_w = new_w
            st.session_state.display_image = generate_image(G, new_w)
            st.session_state.redo_stack.clear()

# ------------------------- Style Mixing -------------------------
st.markdown("### 🔀 Style Mixing")
col_mix = st.columns([2, 2, 1])

with col_mix[0]:
    if st.button("🎲 Sample Second Face"):
        z2 = torch.randn(1, G.z_dim)
        w2 = G.mapping(z2, None)
        st.session_state.secondary_w = w2.unsqueeze(1).repeat(1, G.num_ws, 1)

with col_mix[1]:
    crossover_point = st.slider("Crossover Layer", 1, G.num_ws - 1, G.num_ws // 2)

with col_mix[2]:
    if st.button("🧬 Mix Faces") and st.session_state.secondary_w is not None and st.session_state.current_w is not None:
        st.session_state.undo_stack.append(st.session_state.current_w.clone())
        mixed = style_mix(st.session_state.current_w, st.session_state.secondary_w, crossover_point)
        st.session_state.current_w = mixed
        st.session_state.display_image = generate_image(G, mixed)
        st.session_state.redo_stack.clear()
