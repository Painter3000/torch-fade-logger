#!/bin/bash
# fade_hip_logger_full_setup.sh
# 🧠 Kombiniertes FADE Logger-Integrations- und Instrumentierungsskript für PyTorch 2.9 mit ROCm/HIP

set -e
echo "🔍 Starte vollständige FADE Logger Integration und Instrumentierung..."
echo ""

# Schritt 1: Logger-Header erstellen
mkdir -p ./c10/hip/
cat > ./c10/hip/fade_logger.h << 'EOF'
// ./pytorch/c10/hip/fade_logger.h - FADE-Logger-Header
#ifndef FADE_LOGGER_H
#define FADE_LOGGER_H

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>

class FADELogger {
private:
    static std::mutex log_mutex;
    static std::ofstream log_file;
    static bool initialized;
    
public:
    static void init() {
        if (!initialized) {
            log_file.open("/workspace/fade_pytorch_debug.jsonl", std::ios::app);
            initialized = true;
        }
    }
    
    static void log(const std::string& function, const std::string& message, 
                   int device_id = -1, const std::string& extra = "") {
        std::lock_guard<std::mutex> lock(log_mutex);
        init();
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << "{"
           << "\"timestamp\":\"" << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S") << "\","
           << "\"function\":\"" << function << "\","
           << "\"message\":\"" << message << "\"";
        
        if (device_id >= 0) {
            ss << ",\"device_id\":" << device_id;
        }
        if (!extra.empty()) {
            ss << ",\"extra\":\"" << extra << "\"";
        }
        
        ss << "}";
        log_file << ss.str() << std::endl;
    }
};

#define FADE_LOG(function, message, ...) FADELogger::log(function, message, ##__VA_ARGS__)

#endif // FADE_LOGGER_H
EOF

echo "✅ Logger-Header erstellt."

# Schritt 2: fade_logger.cpp erstellen
cat > ./c10/hip/fade_logger.cpp << 'EOF'
// ./pytorch/c10/hip/fade_logger.cpp - FADE-Logger
#include "fade_logger.h"

std::mutex FADELogger::log_mutex;
std::ofstream FADELogger::log_file;
bool FADELogger::initialized = false;
EOF

echo "✅ fade_logger.cpp erstellt."
echo ""

# Schritt 3: c10/hip/CMakeLists.txt patchen
sed -i.bak '/file(GLOB C10_HIP_SRCS/,/impl\/\*\.cc/{/impl\/\*\.cc/ a\
        fade_logger.cpp
}' c10/hip/CMakeLists.txt
echo "🔧 CMakeLists.txt angepasst: fade_logger.cpp eingebunden."

grep -q 'target_compile_options(c10_hip PRIVATE "-fvisibility=default")' c10/hip/CMakeLists.txt || \
sed -i '/target_compile_options(c10_hip PRIVATE "-fvisibility=hidden")/a\
        target_compile_options(c10_hip PRIVATE "-fvisibility=default")' c10/hip/CMakeLists.txt
echo "🔧 CMakeLists.txt angepasst: -fvisibility=default eingebunden."
echo ""

# Schritt 4: Automatische Instrumentierung starten
echo "🔍 Starte FADE-Instrumentierung für HIP-relevante Module..."
echo ""

declare -A FADE_CATEGORIES=(
    ["hipMalloc"]="MEMORY"
    ["hipFree"]="MEMORY"
    ["hipMemcpy"]="MEMORY"
    ["hipSetDevice"]="DEVICE"
    ["hipGetDeviceCount"]="DEVICE"
    ["hipGetDeviceProperties"]="DEVICE"
    ["hipStreamCreate"]="STREAM"
    ["hipStreamDestroy"]="STREAM"
    ["hipEventCreate"]="STREAM"
    ["hipEventRecord"]="STREAM"
    ["hipStreamWaitEvent"]="STREAM"
    ["cuda.*hip"]="REDIRECT"
    ["CUDA.*HIP"]="REDIRECT"
)

INCLUDE_PATHS=(
    "aten/src/ATen/hip/"
    "aten/src/ATen/native/"
    # "c10/hip/"
    "c10/core/"
    "torch/csrc/cuda/"
    "aten/src/ATen/hip/impl/"
)

for path in "${INCLUDE_PATHS[@]}"; do
    find "$path" -name "*.cpp" -type f | while read -r file; do
        [[ "$file" =~ \.fade_backup$ ]] && continue
        cp "$file" "$file.fade_backup"

        if ! grep -q 'fade_logger.h' "$file"; then
            sed -i '1i#include "c10/hip/fade_logger.h"' "$file"
        fi

        for pattern in "${!FADE_CATEGORIES[@]}"; do
            matches=$(grep -En "$pattern" "$file" | cut -d: -f1)
            for lineno in $matches; do
                pre_line=$(sed -n "$((lineno-1))p" "$file")
                if echo "$pre_line" | grep -qE '\)\s*\{'; then
                    tag="${FADE_CATEGORIES[$pattern]}"
                    sed -i "$((lineno+1))i\\\nFADE_LOG(\"$tag\", \"$pattern detected at line $lineno in $(basename $file)\");" "$file"
                    echo "✅ FADE_LOG hinzugefügt ($tag) in $file:$lineno"
                else
                    echo "⚠️  FADE_LOG übersprungen: Kein Funktionskontext ($file:$lineno)"
                fi
            done
        done
    done
done

echo ""
echo "🎉 FADE Logger Integration und Instrumentierung erfolgreich abgeschlossen."
echo ""

# Analyse-Skript erstellen
cat > fade_log_analyzer.py << 'EOF'
#!/usr/bin/env python3
# fade_log_analyzer.py

"""
FADE PyTorch Logger Analyzer — analysiert FADE Debug-Logs für ROCm/HIP Performance und Kompatibilität
"""

import json
import sys
import os
from collections import defaultdict, Counter
from datetime import datetime
import argparse
from shutil import move

def parse_args():
    parser = argparse.ArgumentParser(
        description="FADE PyTorch Logger Analyzer — ROCm/HIP Debug-Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("-f", "--file", default="/workspace/fade_pytorch_debug.jsonl",
                        help="Pfad zur Logdatei (.jsonl), Standard: /workspace/fade_pytorch_debug.jsonl")
    parser.add_argument("--verbose", action="store_true",
                        help="Zeige vollständige Event-Daten")
    parser.add_argument("--archivieren", action="store_true",
                        help="Archiviert die Logdatei nach der Analyse")
    parser.add_argument("--time", action="store_true",
                        help="Zeigt zeitliche Verteilung der Events (nach Stunde)")
    parser.add_argument("--speicher", action="store_true",
                        help="Zeigt Speicheraktivitäten wie malloc/free")
    parser.add_argument("--geräte", action="store_true",
                        help="Zeigt Funktionsaufrufe pro Gerät")

    return parser.parse_args()

def analyze_fade_log(log_file, verbose=False, show_time=False, show_mem=False, show_devices=False, archivieren=False):
    if not os.path.exists(log_file):
        print(f"❌ Log-Datei nicht gefunden: {log_file}")
        return

    events = []
    device_usage = Counter()
    function_calls = Counter()

    with open(log_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                events.append(event)
                if 'device_id' in event:
                    device_usage[event['device_id']] += 1
                function_calls[event.get('function', 'UNKNOWN')] += 1
                if verbose:
                    print(event)
            except json.JSONDecodeError:
                continue

    print("🧠 FADE PyTorch Debug-Analyse")
    print("=" * 50)
    print(f"📊 Gesamt-Events: {len(events)}")
    print(f"🎯 Aktive Geräte: {dict(device_usage)}")

    print("\n📋 Top Funktionsaufrufe:")
    for func, count in function_calls.most_common(10):
        print(f"   {func}: {count}")

    if show_mem:
        alloc_ops = [e for e in events if "malloc" in e.get("function", "").lower()]
        free_ops = [e for e in events if "free" in e.get("function", "").lower()]
        print(f"\n🧮 GPU Mallocs: {len(alloc_ops)}")
        print(f"🗑️ GPU Frees: {len(free_ops)}")

    redirect_ops = [e for e in events if "CUDA_TO_HIP" in e.get("function", "")]
    print(f"\n🔄 CUDA→HIP Weiterleitungen: {len(redirect_ops)}")

    if show_time:
        time_buckets = defaultdict(int)
        for event in events:
            timestamp = event.get("timestamp")
            if timestamp:
                try:
                    t = datetime.fromisoformat(timestamp)
                    bucket = t.strftime("%H:%M")
                    time_buckets[bucket] += 1
                except ValueError:
                    continue
        print("\n⏱️ Event-Zeiten:")
        for time, count in sorted(time_buckets.items()):
            print(f"   {time}: {count} Events")

    if show_devices:
        print("\n🎛️ Funktionsaufrufe pro Gerät:")
        device_func = defaultdict(Counter)
        for e in events:
            dev = e.get("device_id")
            func = e.get("function")
            if dev is not None and func:
                device_func[dev][func] += 1
        for dev_id, funcs in device_func.items():
            print(f"  📦 GPU {dev_id}:")
            for func, count in funcs.most_common(5):
                print(f"    {func}: {count}")

    if archivieren:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archiv_file = f"/workspace/log_archive/fade_log_{ts}.jsonl"
        os.makedirs(os.path.dirname(archiv_file), exist_ok=True)
        move(log_file, archiv_file)
        print(f"\n📦 Log archiviert unter: {archiv_file}")

def main():
    args = parse_args()

    # Falls keine Argumente übergeben wurden, Standardanalyse starten
    if len(sys.argv) == 1:
        print("ℹ️ Keine Optionen übergeben — Standardanalyse wird ausgeführt...")
        print("ℹ️ Für weitere Optionen: python3 fade_log_analyzer.py --help \n")

    analyze_fade_log(
        log_file=args.file,
        verbose=args.verbose,
        show_time=args.time,
        show_mem=args.speicher,
        show_devices=args.geräte,
        archivieren=args.archivieren
    )

if __name__ == "__main__":
    main()
EOF

chmod +x fade_log_analyzer.py
echo "✅ fade_log_analyzer.py erstellt."
echo ""

echo "📋 Nächste Schritte:"
echo "   1. PyTorch builden: ./build_rocm_v2.9.sh"
echo "   2. Logs analysieren: python3 fade_log_analyzer.py (-h / --help)"
echo "   3. Debug-Log ansehen: tail -f /workspace/fade_pytorch_debug.jsonl"
echo ""
echo "🔍 Logger-Features:"
echo "   - GPU Device Tracking"
echo "   - Memory Allocation Monitoring" 
echo "   - CUDA→HIP Redirect Detection"
echo "   - Stream Operations Logging"
echo "   - JSONL Format für maschinelle Auswertung"
