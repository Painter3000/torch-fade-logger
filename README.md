# torch-fade-logger

Runtime GPU Validator for PyTorch on ZLUDA/ROCm

* Linux Mint 21.2 – Victoria / Ubuntu 22.04 LTS
* Hardware Reference: AMD RX 6800 XT (RDNA2)
* Docker workflow with post-installation of dev packages
* Correct build order: `build_amd.py` **before** logger patch
* Install FADELogger and/or only the Patch (fade_amd_cu_fix.sh)
* CMake customization in `build_rocm_v2.9.sh`

---

📂 Directory structure:

```bash
./workspace/
    ├── Dockerfile
    └── pytorch
          ├── build_rocm_v2.9.sh
          ├── fade_amd_cu_fix.sh
          ├── fade_amd_cu_fix_README-DE.md
          ├── fade_amd_cu_fix_README-ENG-US.md
          └── fade_hip_logger_full_setup.sh
```

💡 All required files and subfolders will be generated automatically via script!

---

# 🔦 FADELogger – Runtime GPU Validation for ZLUDA, ROCm & PyTorch

> Tested on: AMD Radeon RX 6800 XT (RDNA2) using Docker and ROCm 6.4.x
> Supports Torch 2.9 Dev (branch `main`, as of July 2025)

---

## 🎯 Motivation

As of version 2.9, PyTorch still provides **very limited or no real support** for AMD GPUs – CPU fallback is often the norm.
**ZLUDA**, originally developed for Intel, allows CUDA workloads to run on AMD GPUs via HIP, essentially faking "GPU support" for Torch.

But this emulation is not without issues:

🧠 For example, an RX 6800 XT physically has **72 Compute Units**, but ZLUDA only detects **36**.
These incorrect device properties lead to distorted performance and faulty resource allocation – often unnoticed by developers.

**FADELogger** addresses this directly:

* Logs HIP and Torch runtime calls
* Detects incorrect GPU metadata – e.g. CU count, clock speed, architecture
* Optionally returns corrected values
  👉 E.g. `ComputeUnits=72` instead of `36` on RX 6800 XT

This helps not only verify whether Torch is "pretending" to run on a CUDA device – but also whether the **underlying symbol routing** is functioning correctly.

💡 The result is not just a log file, but a **tool for active validation and correction of GPU behavior on RDNA2 systems**.

---

## 📦 Docker Setup for FADELogger (RDNA2)

> Tested with: AMD RX 6800 XT (gfx1030), ROCm 6.4.x, Ubuntu 22.04

The following Dockerfile sets up a full development environment for:

* 🧱 Building PyTorch 2.9 with ROCm
* 🔬 Integrating FADELogger (C++ + Python)
* ⚙️ HIP runtime debugging on RDNA2

### 🔧 Dockerfile (example)

```dockerfile
# PyTorch ROCm Build Container for AMD RX 6800 XT
FROM rocm/dev-ubuntu-22.04:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_ROCM_ARCH=gfx1030
ENV USE_ROCM=1
ENV USE_CUDA=0
ENV MAX_JOBS=4

RUN apt-get update && apt-get install -y \
  apt-utils \
  hipsolver-dev \
  rocthrust-dev \
  hipcub-dev \
  rocprim-dev \
  hipsparse-dev \
  hipfft-dev \
  hipblas-dev \
  rocblas-dev \
  miopen-hip-dev \
  hiprand-dev \
  rocrand-dev \
  rccl-dev \
  rocfft-dev \
  python3 \
  python3-pip \
  python3-dev \
  python3-setuptools \
  git cmake ninja-build build-essential \
  libopenblas-dev libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libatlas-base-dev gfortran pkg-config \
  software-properties-common wget curl \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install \
  numpy \
  pyyaml \
  typing_extensions \
  requests \
  future \
  six \
  cmake>=3.27 \
  ninja \
  packaging

WORKDIR /workspace
ENV PATH="/opt/rocm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"
CMD ["/bin/bash"]
```

---

### ▶️ Build container from Dockerfile

```bash
docker build -t pytorch-fade .
```

### ▶️ Run container with extended permissions

```bash
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v /home/oem/Programme/docker/FADE:/workspace \
  pytorch-rocm-build
```

🧩 **What these parameters do:**

| Parameter                           | Purpose                                                  |
| ----------------------------------- | -------------------------------------------------------- |
| `--device=/dev/kfd`                 | Access to ROCm kernel fusion dispatcher                  |
| `--device=/dev/dri`                 | Access to Direct Rendering Infrastructure (GPU access)   |
| `--group-add video`                 | Adds container to video group – required for GPU usage   |
| `--cap-add=SYS_ADMIN`               | Grants admin capabilities (e.g. for debugging or mounts) |
| `--cap-add=SYS_PTRACE`              | Allows debugging inside the container                    |
| `--security-opt seccomp=unconfined` | Disables syscall filtering – more flexibility            |
| `--ipc=host`                        | Shares IPC memory with host – useful for PyTorch         |
| `-v ...:/workspace`                 | Mounts host project folder into container                |
| `-it`                               | Interactive terminal                                     |

✨ These settings ensure the container is **fully GPU-ready for deep learning** with ROCm and FADELogger.

---

### 📌 Notes

* Ensure that dev packages like `rocblas-dev`, `hipfft-dev` etc. are present, otherwise PyTorch will fail to build!
* `PYTORCH_ROCM_ARCH=gfx1030` is tailored for **RX 6800 XT (RDNA2)** – adjust as needed for other GPUs (e.g. `gfx1100`)
* You can `git clone` PyTorch in the Dockerfile, but it's often cleaner to do it manually for forks/branches.

💡 Useful Docker commands:

```bash
docker ps -a
```

```bash
docker start -ai <CONTAINER-NAME>
# e.g. docker start -ai tender_mahavira
```

---

### 2. Post-install dev packages (if needed)

Sometimes ROCm Docker images don't fully install the dev packages, even when listed in the Dockerfile. Run this inside the container:

```bash
apt-get update && apt-get install -y \
  hiprand-dev rocblas-dev miopen-hip-dev \
  rocrand-dev rccl-dev rocfft-dev \
  hipblas-dev hipfft-dev hipsparse-dev \
  rocprim-dev hipcub-dev rocthrust-dev \
  hipsolver-dev hipsparselt-dev
```

---

## ⚙️ PyTorch (2.9 Dev) Setup

### 1. Clone source

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

### 2. Generate HIP-specific files

```bash
python3 tools/amd_build/build_amd.py
```

> ⚠️ **Important:** This script generates HIP-related files (e.g. `HIPGuardImplMasqueradingAsCUDA.*`)
> that are **required for FADELogger instrumentation**.
> Running the logger patch **before this** will result in missing hooks!

---

## 🚀 Install FADELogger and / or only the Patch

```bash
./fade_hip_logger_full_setup.sh
```

**What this script does:**

* 📄 Generates logger files (`fade_logger.cpp`, headers)
* 🧩 Patches CMakeLists.txt
* 🧠 Adds hooks for `hipMalloc`, `hipLaunchKernel`, `hipMemcpy`, etc.
* 🔍 Optionally creates log analyzer script: `fade_log_analyzer.py`

**The AMD-CU-fix-Patch**

```bash
./fade_amd_cu_fix.sh
```
**What this Patch does:**

* ✨ Our patch intercepts the HIP device property detection and corrects the `multiProcessorCount` for affected GPUs.

---

## 🛠️ Build with ROCm

```bash
./build_rocm_v2.9.sh
```

> 📝 Edit `build_rocm_v2.9.sh` as needed:
>
> * Set ROCm paths (e.g. `--rocm-path`)
> * Set GPU architecture (`gfx1030`, etc.)
> * Add/remove CMake flags

After build, check version:

```bash
pip list | grep torch
# → torch 2.9.0a0+gXXXXX /workspace/pytorch
```

---

## 🧪 Run FADELogger

```bash
python3 fade_log_analyzer.py
```

Example output:

```
🧠 FADE PyTorch Debug Analysis
==============================
📊 Total Events: 44
🎯 Active Devices: {0: 1}

📋 Top Calls:
REDIRECT: 32
DEVICE: 8
TEST: 2
STREAM: 1
```

---

## 📊 Feature Overview

| Category        | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| 🎯 Device Check | Logs `hipGetDeviceProperties`, CU count, arch, clock rate           |
| 🔄 Redirects    | Detects CUDA→HIP translation via symbol interception (e.g. ZLUDA)   |
| 🧠 Function Log | Hooks `hipLaunchKernel`, `hipMemcpy`, `hipMalloc`, etc.             |
| 🧾 JSONL Output | Timestamps, functions, arguments, return codes – ideal for analysis |

---

## ⚠️ Why use FADELogger?

* PyTorch 2.9 often **fails to detect HIP/ROCm properly**, even when `libamdhip64.so` is loaded
* ZLUDA may **spoof device properties** (e.g. wrong CU count)
* Torch often **silently falls back to CPU**, with no warning
* Many calls **aren’t executed at all**, yet no error is raised

FADELogger helps you answer:

* 🔍 Was `hipLaunchKernel()` actually called?
* 💥 Was GPU memory allocated via `hipMalloc()`?
* ⛔ Were redirects or errors intercepted?

---

## 🛠️ ToDo / Extensions

* [ ] Hook additional HIP APIs (e.g. `hipGraphLaunch`, `hipStreamWaitEvent`)
* [ ] Diagram showing the data flow (CUDA → ZLUDA → HIP → FADE)
* [ ] Optional integration with `strace`, `dlopen` tracer, or symbol debugger

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE)  
© 2025 Painter3000 (michael.w.kuebel@web.de)

---

## 🤝 Acknowledgements

This project was made possible through the combined power of:

- 🤖 **ChatGPT (GPT-4o)** – for detailed architecture help, bug hunting, and real-time debugging
- 🧠 **GPT-4.5** – for background research, documentation summarization, and advanced technical insight
- 💼 **Microsoft Copilot** – for code context integration and productivity suggestions
- 🧬 **Claude** – for occasional sanity checks and alternative API ideas

Thank you to all the AI systems that supported the development of FADELogger.  
You're not just tools – you're true digital collaborators. 😊
