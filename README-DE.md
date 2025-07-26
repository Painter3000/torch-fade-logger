# torch-fade-logger
Runtime GPU Validator for PyTorch on ZLUDA/ROCm

* Linux Mint 21.2 - Victoria / Ubuntu 22.04 LTS 
* Hardware-Referenz (RDNA2 RX 6800 XT) 
* Docker-Workflow inkl. Nachinstallation der Dev-Pakete 
* richtigen Ablauf zwischen `build_amd.py` und Logger-Patch 
* CMake-Anpassung in `build_rocm_v2.9.sh`
---

📂 Logger-Verzeichnisstruktur:
```bash
./workspace/
    ├── Dockerfile
    └── pytorch
          ├── build_rocm_v2.9.sh
          └── fade_hip_logger_full_setup.sh
```
💡 Alle benötigten Dateien, sowie dazugehörige Ordner werden mittels Skript erzeugt!

--- 

# 🔦 FADELogger – Runtime-GPU-Validierung für ZLUDA, ROCm & PyTorch 

> Getestet auf: AMD Radeon RX 6800 XT (RDNA2) unter Docker mit ROCm 6.4.x 
> Unterstützt Torch 2.9 Dev (Branch `main`, Stand: Juli 2025) 

# 🎯 Motivation 

PyTorch unterstützt AMD-GPUs selbst bei der Version 2.9 **nur rudimentär bzw. gar nicht** – GPU-Fallbacks zur CPU sind leider häufig die Regel. 
**ZLUDA**, ursprünglich für Intel entwickelt, simuliert CUDA auf AMD über HIP – bietet also eine Möglichkeit, Torch „GPU-Unterstützung“ als Nvidia-GPU vorzutäuschen. 


Doch diese Simulation hat Tücken: 

🧠 Eine RX 6800 XT besitzt z. B. **72 Compute Units**, ZLUDA erkennt jedoch nur **36**. 
Diese falschen Device-Werte führen zu verzerrter Performance und fehlerhafter Ressourcenallokation – ohne dass der Entwickler es merkt. 

Der **FADELogger** setzt genau hier an: 

- Er dokumentiert HIP- und Torch-Aufrufe **zur Laufzeit** 
- Er erkennt falsche GPU-Metadaten – wie CU-Zahl, Taktung, Architektur 
- Er bietet die Option, korrigierte Werte zurückzugeben 
  👉 z. B. `ComputeUnits=72` statt `36` bei RX 6800 XT 

Damit wird nicht nur sichtbar, ob Torch **angeblich** auf einer CUDA-GPU läuft – sondern auch **wie korrekt** die zugrunde liegende Symbol-Logik funktioniert. 

💡 Das Ergebnis ist keine bloße Log-Datei, sondern ein Werkzeug zur aktiven Verifikation und Korrektur der GPU-Schicht auf RDNA2-Systemen. 

--- 

## 📦 Docker-Setup für FADELogger (RDNA2)

> Getestet auf: AMD Radeon RX 6800 XT (gfx1030), ROCm 6.4.x, Ubuntu 22.04 Das folgende Dockerfile erzeugt eine vollständige Entwicklungsumgebung für:

* 🧱 Build von PyTorch 2.9 mit ROCm
* 🔬 FADELogger-Integration (C++ + Python)
* ⚙️ HIP Runtime Debugging auf RDNA2

### 🔧 Dockerfile (Beispiel)
```dockerfile
# PyTorch ROCm Build Container für AMD RX 6800 XT
FROM rocm/dev-ubuntu-22.04:latest

# Umgebungsvariablen setzen
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_ROCM_ARCH=gfx1030
ENV USE_ROCM=1
ENV USE_CUDA=0
ENV MAX_JOBS=4

# System-Updates und Abhängigkeiten installieren
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
  libopenblas-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libv4l-dev \
  libatlas-base-dev \
  gfortran pkg-config \
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


### ▶️ Container mit Dockerfile erzeugen (lokal)
```bash
docker build -t pytorch-fade .
```

### ▶️ Container mit Parameter starten (lokal)
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

Damit wird der Container nicht nur einfach gestartet, sondern bekommt gezielt Zugriffe und Rechte, die für Hardwarebeschleunigung (z. B. über ROCm) und Debugging wichtig sind. Hier ist, was die einzelnen Parameter bewirken:  

🧩 **Parameter-Erklärung**:

| Parameter                         | Zweck                                                                 |
|----------------------------------|------------------------------------------------------------------------|
| `--device=/dev/kfd`              | Zugriff auf die ROCm-Komponente für Kernel Fusion Dispatcher          |
| `--device=/dev/dri`              | Ermöglicht Zugriff auf Direct Rendering Infrastructure (GPU-Zugriff)  |
| `--group-add video`              | Fügt den Container der Video-Gruppe hinzu – nötig für GPU-Nutzung     |
| `--cap-add=SYS_ADMIN`            | Erlaubt administrative Funktionen, z. B. für Debugging oder Mounts     |
| `--cap-add=SYS_PTRACE`           | Ermöglicht Debugging im Container durch `ptrace`                      |
| `--security-opt seccomp=unconfined` | Hebt die standardmäßige Syscall-Filterung auf – für mehr Flexibilität |
| `--ipc=host`                     | Container teilt IPC-Speicher mit dem Host – nützlich für PyTorch      |
| `-v /home/...:/workspace`        | Bind-Mount: Host-Verzeichnis wird im Container eingebunden            |
| `-it`                            | Interaktives Terminal                                                 |
| `pytorch-rocm-build`             | Das verwendete Image                                                  |

✨ Dadurch wird der Container **performancefähig für GPU-gestützte Deep Learning-Anwendungen** wie FADE – besonders wenn du mit ROCm arbeitest.  

### 📌 Hinweise 

* **Entwicklerpakete (dev)** wie `rocblas-dev`, `hipfft-dev` usw. müssen **zwingend vorhanden sein**, sonst scheitert der Torch-Build! 
* `PYTORCH_ROCM_ARCH=gfx1030` ist angepasst für **RX 6800 XT (RDNA2)**. Für andere Karten ggf. `gfx1031`, `gfx1100`, etc. wählen. 
* Optional kann im Dockerfile direkt `git clone https://github.com/pytorch/pytorch` eingefügt werden – aber besser manuell ausführt, falls Forks oder Branches genutzt werden. 

💡Nützliche Docker- Befehle aus Host-Terminal: 
```bash 
docker ps -a
```
- Container auflisten (alle)

```bash 
docker start -ai <CONTAINER-NAME>
```
- starten des Docker-Containers z.B. docker start -ai tender_mahavira
--- 

### 2. Dev-Pakete **nachträglich installieren** bei Bedarf

Das ROCm-Docker-Image enthält manch mal nicht alle nötigen Development-Pakete, obwohl diese zuvor im Dockerfile angegeben wurden! 

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

### 1. Klonen
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
``` 
### 2. ROCm-spezifische HIP-Dateien generieren
```bash
python3 tools/amd_build/build_amd.py
```
> ⚠️ **Wichtig:** Dieses Skript erzeugt HIP-spezifische Dateien (u. a. `HIPGuardImplMasqueradingAsCUDA.*`), 
> die für die FADELogger-Integration zwingend notwendig sind! 
> Wer direkt mit dem Logger-Patch beginnt, wird viele Stellen 
**nicht instrumentiert bekommen**. 

--- 

## 🚀 FADELogger installieren 
```bash 
./fade_hip_logger_full_setup.sh 
``` 

**Was passiert dabei?** 

* 📄 Logger-Dateien (`fade_logger.cpp`, Header) werden erstellt 
* 🧩 CMakeLists.txt wird automatisch erweitert 
* 🧠 Instrumentierung für `hipMalloc`, `hipLaunchKernel`, `hipMemcpy`, `hipStreamSynchronize`, ... 
* 🔍 Optional: Log-Auswertungsskript `fade_log_analyzer.py` wird erzeugt 

--- 


## 🛠️ Build mit ROCm 
```bash 
./build_rocm_v2.9.sh 
``` 

> 📝 Passe in dieser Datei (`build_rocm_v2.9.sh`) bei Bedarf: 
> 
> * die Pfade zu ROCm (z. B. `--rocm-path`) 
> * die GPU-Architektur (`gfx1030` o. ä.) 
> * und weitere CMake-Flags an 

Nach erfolgreichem Build prüfen: 
```bash 
pip list | grep torch # → torch 2.9.0a0+gXXXXX /workspace/pytorch 
``` 

--- 

## 🧪 FADELogger ausführen 
```bash 
python3 fade_log_analyzer.py 

``` 

Beispielausgabe: 

```
🧠 FADE PyTorch Debug-Analyse 
============================= 
📊 Gesamt-Events: 44 
🎯 Aktive Geräte: {0: 1} 

📋 Top Funktionsaufrufe: 
REDIRECT: 32 
DEVICE: 8 
TEST: 2 
STREAM: 1
```

---

## 📊 Feature-Übersicht 

| Kategorie | Beschreibung | 
| --------------- | --------------------------------------------------------------------------- | 
| 🎯 Device-Check | Loggt `hipGetDeviceProperties`, CU-Anzahl, Architektur, Taktrate | 
| 🔄 Redirects | Erkennt CUDA→HIP-Umleitungen (Symbol-Interception, ZLUDA-Bypass) | 
| 🧠 Funktionslog | Dokumentiert Aufrufe wie `hipLaunchKernel`, `hipMemcpy`, `hipMalloc` | 
| 🧾 JSONL-Format | Zeitstempel, Funktion, Argumente, Ergebnis – ideal für automatische Analyse | 

--- 

## ⚠️ Warum FADELogger? 

* PyTorch 2.9 erkennt ROCm/HIP oft **nicht korrekt**, obwohl `libamdhip64.so` geladen ist 
* ZLUDA kann **DeviceProperties faken** 
* Torch tendiert zu **silent Fallbacks** → es läuft plötzlich auf CPU 
* Viele Calls werden **gar nicht ausgeführt**, obwohl keine Fehlermeldung erscheint 

Der FADELogger zeigt dir in Echtzeit: 

* 🔍 Wurde `hipLaunchKernel()` wirklich aufgerufen? 
* 💥 Wurde Speicher via `hipMalloc()` korrekt alloziert? 
* ⛔ Wurden Redirects oder Funktionsfehler registriert? 

--- 

## 🛠️ ToDo / Erweiterungen 
* [ ] Integration zusätzlicher Hooks (`hipGraphLaunch`, `hipStreamWaitEvent`) 
* [ ] Diagramm zur Datenflusskette (CUDA→ZLUDA→HIP→FADE) 
* [ ] Optionale Verbindung zu `strace`, `dlopen`-Tracer oder Symbol-Debugger 
---
