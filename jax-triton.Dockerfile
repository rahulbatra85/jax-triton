
FROM rocm/jax:rocm5.5.0-jax0.4.12-py3.10.0

# Triton depends on PyTorch
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.5

# Fetch the Triton dev bits for ROCm and install
# RUN cd ~ && git clone https://github.com/ROCmSoftwarePlatform/triton.git
# RUN git clone -b FA_triton_benchmark https://github.com/ROCmSoftwarePlatform/flash-attention.git
# RUN cd ~/triton/python && pip install -v -e .

RUN pip install pandas pytest

# bazel
RUN apt install apt-transport-https curl gnupg -y
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
RUN mv bazel-archive-keyring.gpg /usr/share/keyrings
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
RUN apt update && apt install -y bazel
