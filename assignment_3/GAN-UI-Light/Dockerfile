FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

# Download latent direction vectors into /app/presets
RUN mkdir -p /app/presets && \
    wget https://github.com/genforce/interfacegan/raw/master/boundaries/eyeglass_boundary.npy -O /app/presets/glasses.npy && \
    wget https://github.com/genforce/interfacegan/raw/master/boundaries/beard_boundary.npy -O /app/presets/beard.npy && \
    wget https://github.com/genforce/interfacegan/raw/master/boundaries/age_boundary.npy -O /app/presets/age.npy && \
    wget https://github.com/genforce/interfacegan/raw/master/boundaries/gender_boundary.npy -O /app/presets/gender.npy && \
    wget https://github.com/genforce/interfacegan/raw/master/boundaries/smile_boundary.npy -O /app/presets/smile.npy && \
    wget https://github.com/genforce/interfacegan/raw/master/boundaries/child_boundary.npy -O /app/presets/child.npy && \
    wget https://raw.githubusercontent.com/justinpinkney/stylegan2-ada-transformer/main/directions/cartoon.npy -O /app/presets/cartoon.npy && \
    wget https://raw.githubusercontent.com/justinpinkney/stylegan2-ada-transformer/main/directions/artistic.npy -O /app/presets/artistic.npy




RUN pip3 install -r requirements.txt

# Clone StyleGAN2-ADA-PyTorch and install requirements
RUN git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git /app/stylegan2-ada-pytorch
WORKDIR /app/stylegan2-ada-pytorch

# Install any specific requirements (usually just torch, torchvision, click)
RUN pip install imageio-ffmpeg==0.4.3 pyspng==0.1.0

# Download pre-trained weights
RUN wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -P pretrained/

# Back to root directory for streamlit launch
WORKDIR /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]