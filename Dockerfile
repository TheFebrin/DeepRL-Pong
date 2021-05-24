FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
    && apt-get install -y \
    build-essential \
    python3.8 \
    python3-pip \
    python3-dev \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    ffmpeg \
    curl \
    patchelf \
    libglfw3 \
    libglfw3-dev \
    cmake \
    zlib1g \
    zlib1g-dev \
    swig \
    wget \
    unrar \
    unzip \
    && pip3 -q install pip --upgrade
WORKDIR src/
COPY . .
RUN pip3 install -r requirements.txt
RUN wget http://www.atarimania.com/roms/Roms.rar \
    && unrar x Roms.rar \
    && unzip ROMS.zip \
    && python3 -m atari_py.import_roms ROMS
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]