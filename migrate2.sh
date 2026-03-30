#!/usr/bin/env bash
# Continue migration, skip FLUX models
set -euo pipefail

COMFYUI_DIR="/mnt/e/ComfyUI/models/checkpoints"
GPU_BROKER="/home/lyweiwei/projects/gpu-broker"
VENV="$GPU_BROKER/.venv/bin/python"
LOG="$GPU_BROKER/migrate2.log"

# FLUX models to skip (too big for 12GB VRAM)
SKIP="getphatFLUXReality_v11Softcore|majicflus_v10"

cd "$GPU_BROKER"
total=0; success=0; skipped=0; failed=0

echo "=== Migration Phase 2 (skip FLUX) ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

for f in "$COMFYUI_DIR"/*.safetensors; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .safetensors)
    
    if echo "$name" | grep -qE "^($SKIP)$"; then
        echo "⏭️ Skipping FLUX model: $name" | tee -a "$LOG"
        skipped=$((skipped + 1))
        continue
    fi
    
    total=$((total + 1))
    size=$(du -h "$f" | cut -f1)
    echo "[$total] Moving $name ($size)..." | tee -a "$LOG"
    
    if $VENV -m gpu_broker.cli model add "$f" --move --name "$name" 2>&1 | tee -a "$LOG"; then
        success=$((success + 1))
        echo "  ✅ Success" | tee -a "$LOG"
    else
        failed=$((failed + 1))
        echo "  ❌ Failed" | tee -a "$LOG"
    fi
done

echo "=== Done ===" | tee -a "$LOG"
echo "Total: $total | Success: $success | Skipped: $skipped | Failed: $failed" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
