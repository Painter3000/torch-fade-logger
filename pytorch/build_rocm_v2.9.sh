#!/bin/bash
# build_rocm_v2.9.sh

echo "🧪 Verlinkung mit rocm_smi64 testen..."
echo '#include <rocm_smi/rocm_smi.h>
int main() { rsmi_init(0); return 0; }' > test_rsmi.cpp

g++ test_rsmi.cpp -o test_rsmi -I/opt/rocm/include -L/opt/rocm/lib -lrocm_smi64 \
  && echo "✅ rsmi_init erfolgreich gelinkt" \
  || { echo "❌ rsmi_init NICHT gefunden – prüfe -I und -L Pfade!"; exit 1; }

rm test_rsmi.cpp test_rsmi

# Exit bei Fehlern
set -e

echo "🔧 Konfiguriere ROCm PyTorch-Build..."

# Setze Library-Path für Laufzeit-Linking (z. B. rsmi_init)
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm-6.4.1/lib:$LD_LIBRARY_PATH

# ROCm-Umgebungsvariablen
export PYTORCH_ROCM_ARCH=gfx1030
export USE_ROCM=1
export USE_CUDA=0
export HIP_PLATFORM=amd
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm-6.4.1

# Weitere Build-Optionen
export USE_GLOO=OFF
export USE_NCCL=OFF
export USE_DISTRIBUTED=OFF
export USE_MIOPEN=ON
export MAX_JOBS=8
export CMAKE_BUILD_TYPE=Release
export BUILD_TEST=0
export BUILD_BINARY=0

# Linker-Flags für rsmi_init
export CMAKE_EXE_LINKER_FLAGS="-L/opt/rocm/lib -lrocm_smi64"
export CMAKE_SHARED_LINKER_FLAGS="-L/opt/rocm/lib -lrocm_smi64"
export LDFLAGS="-L/opt/rocm/lib -lrocm_smi64"
export TORCH_CXXFLAGS="-L/opt/rocm/lib -lrocm_smi64"

# Optional: vorherigen Build bereinigen
echo "🧹 Bereinige alten Build..."
rm -rf build/ dist/ torch.egg-info/ .setuptools-cmake-build/ CMakeCache.txt

# Umgebungsvariable anzeigen
echo $PATH
echo $PYTORCH_ROCM_ARCH
echo $USE_ROCM

# Umgebungsvariablen auflisten
printenv | grep ROCM
printenv | grep PATH

# Starte den Build
echo "🚀 Starte PyTorch ROCm-Build..."
python3 setup.py develop

echo "✅ Build abgeschlossen!"

# Prüfen, ob librocm_smi64 korrekt im System vorhanden und verlinkbar ist
echo -e "\n🔎 Prüfe dynamischen Link auf librocm_smi64.so in FADE-Logger..."

if ldd ./build/lib/libc10_hip.so | grep -q "librocm_smi64"; then
  echo "✅ librocm_smi64 ist dynamisch verlinkt in libc10_hip.so"
else
  echo "⚠️ librocm_smi64 ist NICHT direkt verlinkt – evtl. nur indirekt durch fade_logger_test"
fi

# Alternativ: direkter Verifikationstest mit FADE-Testprogramm
echo "🧪 Kompiliere fade_logger_test zur Verlinkungsprüfung..."
echo '#include "c10/hip/fade_logger.h"
int main() { FADE_LOG("TEST", "Logger-Check"); return 0; }' > fade_logger_test.cpp

g++ fade_logger_test.cpp c10/hip/fade_logger.cpp -o fade_logger_test -I. -L/opt/rocm/lib -lrocm_smi64 && \
  echo "✅ fade_logger_test kompiliert und verlinkt erfolgreich gegen librocm_smi64.so" || \
  echo "❌ Fehler beim Kompilieren/Linken – prüfe Pfade und Abhängigkeiten"

rm -f fade_logger_test.cpp fade_logger_test

