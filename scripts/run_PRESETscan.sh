#!/usr/bin/env bash

#A script designed to modify various skies and run them sequentially without having to do so manually.
#To run:
#chmod +x run_PRESETscan.sh
#nohup ./run_PRESETscan.sh > results/logs/scan_master.log 2>&1 & 
# or
# ./scripts/run_PRESETscan.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hemicosmo
set -u

SOUTHS=(1150omc 1178omc 1206omc 1234omc 1262omc)
LOGDIR="results/logs"
mkdir -p "$LOGDIR"

PY="python" 
COMMON="--north fiducial --nside 1024 --delta_l 30 --lmin 32 --apod 1. \
        --blend 3. --beam 0.0 --nsims 1000 --n_threads 30 \
        --phase_mode independent --minuit"

for S in "${SOUTHS[@]}"; do
    LOG="$LOGDIR/asym_fiducial_${S}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== [$(date +%H:%M:%S)] south=$S -> $LOG ==="
    $PY scripts/run_asymmetry.py --south "$S" $COMMON > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "    !! south=$S FAIL (rc=$rc) -- see $LOG ; continue"
    else
        echo "    ok south=$S"
    fi
done

echo "=== scan finished ==="
