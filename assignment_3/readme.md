# Morphix: Real-Time GAN Latent Editor

Morphix is a real-time face generator and editor built using StyleGAN2-ADA and Streamlit. It lets users generate, modify, and mix faces using interpretable latent directions such as Smile, Age, Gender, Beard, Glasses, and more.

---

 Features

- Generate new faces using StyleGAN2-ADA
- Edit attributes using sliders (Smile, Age, Gender)
- One-click transformations (Glasses, Beard, Cartoon)
- Style mixing between two faces
- Undo/Redo edit history
- Download generated image

---

Docker-Based Setup

You can run this app inside a Docker container for full environment reproducibility.
 Step 1: Clone this Repository
```bash
git clone https://github.com/kshitij-2006/SOC-2025/assignment_3/GAN-UI-Light
cd Morphix-GAN-UI
docker build -t morphix-ui .
docker run -p 8501:8501 morphix-ui
to open it 
 http://localhost:8501

