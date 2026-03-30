#!/usr/bin/env bash
# Migrate LoRA models from ComfyUI to gpu-broker
set -euo pipefail

LORA_DIR="/mnt/e/ComfyUI/models/loras"
GPU_BROKER="/home/lyweiwei/projects/gpu-broker"
VENV="$GPU_BROKER/.venv/bin/python"
LOG="$GPU_BROKER/migrate_lora.log"

cd "$GPU_BROKER"
total=0; success=0; failed=0

echo "=== LoRA Migration ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

for f in "$LORA_DIR"/*.safetensors; do
    [ -f "$f" ] || continue
    total=$((total + 1))
    name=$(basename "$f" .safetensors)
    size=$(du -h "$f" | cut -f1)
    
    echo "[$total/44] Moving LoRA: $name ($size)..." | tee -a "$LOG"
    
    if $VENV -m gpu_broker.cli model add "$f" --move --type lora --name "$name" 2>&1 | tee -a "$LOG"; then
        success=$((success + 1))
        echo "  ✅ Success" | tee -a "$LOG"
    else
        failed=$((failed + 1))
        echo "  ❌ Failed" | tee -a "$LOG"
    fi
done

echo "=== LoRA Migration Done ===" | tee -a "$LOG"
echo "Total: $total | Success: $success | Failed: $failed" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
