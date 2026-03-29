#!/usr/bin/env bash
# Batch migrate all ComfyUI checkpoints to gpu-broker (move mode)
# This script: for each .safetensors in ComfyUI, run gpu-broker model add --move
# For already-registered models (like dreamshaper), remove symlink first, then re-add with --move

set -euo pipefail

COMFYUI_DIR="/mnt/e/ComfyUI/models/checkpoints"
GPU_BROKER="/home/lyweiwei/projects/gpu-broker"
BROKER_MODELS="/home/lyweiwei/.gpu-broker/models"
VENV="$GPU_BROKER/.venv/bin/python"
LOG="/home/lyweiwei/projects/gpu-broker/migrate.log"

cd "$GPU_BROKER"

total=0
success=0
skipped=0
failed=0

echo "=== ComfyUI → gpu-broker Migration ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Handle already-registered dreamshaper: remove symlinks, delete DB record, then re-add
if [ -L "$BROKER_MODELS/fdbe56354b8f.safetensors" ]; then
    echo "[PRE] Removing existing dreamshaper symlinks for re-registration..." | tee -a "$LOG"
    rm -f "$BROKER_MODELS/fdbe56354b8f.safetensors"
    rm -f "$BROKER_MODELS/dreamshaperXL_lightningDPMSDE.safetensors"
    # Remove from DB
    $VENV -c "
import sqlite3
conn = sqlite3.connect('/home/lyweiwei/.gpu-broker/gpu-broker.db')
conn.execute(\"DELETE FROM models WHERE id='fdbe56354b8f'\")
conn.commit()
conn.close()
print('DB record removed')
" 2>&1 | tee -a "$LOG"
fi

for f in "$COMFYUI_DIR"/*.safetensors; do
    [ -f "$f" ] || continue
    total=$((total + 1))
    name=$(basename "$f" .safetensors)
    size=$(du -h "$f" | cut -f1)
    
    echo "[$total/33] Moving $name ($size)..." | tee -a "$LOG"
    
    if $VENV -m gpu_broker.cli model add "$f" --move --name "$name" 2>&1 | tee -a "$LOG"; then
        success=$((success + 1))
        echo "  ✅ Success" | tee -a "$LOG"
    else
        exit_code=$?
        if echo "$exit_code" | grep -q "already registered"; then
            skipped=$((skipped + 1))
            echo "  ⏭️ Already registered, skipping" | tee -a "$LOG"
        else
            failed=$((failed + 1))
            echo "  ❌ Failed (exit $exit_code)" | tee -a "$LOG"
        fi
    fi
    echo "" | tee -a "$LOG"
done

echo "=== Migration Complete ===" | tee -a "$LOG"
echo "Total: $total | Success: $success | Skipped: $skipped | Failed: $failed" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
