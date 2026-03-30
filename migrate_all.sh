#!/bin/bash
# Wait for checkpoint migration to finish, then run LoRA migration
echo "Waiting for checkpoint migration (migrate2.sh) to finish..."
while pgrep -f "migrate2.sh" > /dev/null 2>&1; do
    sleep 30
done
echo "Checkpoint migration done! Starting LoRA migration..."
bash /home/lyweiwei/projects/gpu-broker/migrate_lora.sh
