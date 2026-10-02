#!/usr/bin/env bash
#
# H0 scan, two hemispheres:  North = fiducial,  South = 62, 65, 68.5, 71, 74 km/s/Mpc
#
# Every run writes its full log NEXT TO ITS npz:
#   results/fiducial_<S>/asym_fiducial_<S>_<key>[_covisotropic].log
# This script only keeps a short master log (start / end / return code of each run).
# Anything printed BEFORE a run opens its own log (import errors, bad arguments) goes
# to results/logs/launch_<S>_<cov>.txt, which is deleted when the run succeeds.
#
# Pass 1 (stitched covariance) generates all sims; pass 2 (isotropic covariance)
# reuses every cached sim and only refits, so it is cheap.
#
# To run (from anywhere):
#   chmod +x scripts/run_PRESETscan.sh
#   nohup scripts/run_PRESETscan.sh > results/logs/scan_H0_master.log 2>&1 &

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hemicosmo
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."                       # repo root: results/ and cache/ live here

PY="python"
NORTH="fiducial"
SOUTHS=(62H0 65H0 68.5H0 71H0 74H0)         # presets in hemcosmo/config.py
COV_MODES=(stitched isotropic)

COMMON="--north ${NORTH} --nside 1024 --delta_l 30 --lmin 32 --apod 1. \
        --blend 3. --beam 0.0 --nsims 1000 --n_threads 80 \
        --phase_mode independent --minuit \
        --naive_mask_h 6 --naive_mask_v 6 --expected"

# Reuse v1 sim caches (same seeds -> identical sims, but their parameter values cannot be
# verified). Leave empty to regenerate everything. Note: v1 named the 68.5 run '68H0',
# so it cannot be adopted under the new name '68.5H0' and will be regenerated.
ADOPT=""            # or: ADOPT="--adopt_legacy_cache"

LOGDIR="results/logs"
mkdir -p "$LOGDIR"
echo "=== H0 scan started $(date '+%F %T')  N=${NORTH}  S=(${SOUTHS[*]})  cov=(${COV_MODES[*]}) ==="

n_fail=0
for CM in "${COV_MODES[@]}"; do
    EXTRA="--cov_mode ${CM}"
    [ "$CM" = "stitched" ] && EXTRA="${EXTRA} --compare_cov"   # covariance comparison once per S
    for S in "${SOUTHS[@]}"; do
        LAUNCH="${LOGDIR}/launch_${S}_${CM}.txt"
        t0=$(date +%s)
        echo "--- [$(date +%T)] S=${S} cov=${CM}"
        $PY scripts/run_asymmetry.py --south "$S" $COMMON $EXTRA $ADOPT > "$LAUNCH" 2>&1
        rc=$?
        dt=$(( $(date +%s) - t0 ))
        # the run's own log path is printed on its last line ('[log] ... -> path')
        RUNLOG=$(grep -o '\-> .*\.log' "$LAUNCH" | tail -1 | cut -c4-)
        if [ $rc -ne 0 ]; then
            n_fail=$((n_fail + 1))
            echo "    !! FAIL rc=${rc} after ${dt}s  run log: ${RUNLOG:-none}  launch output kept: ${LAUNCH}"
        else
            echo "    ok after ${dt}s  -> ${RUNLOG}"
            rm -f "$LAUNCH"
        fi
    done
done

echo "=== H0 scan finished $(date '+%F %T')  failures: ${n_fail} ==="
