#!/usr/bin/env bash
#
# Single-parameter scans, two hemispheres:  North = fiducial,  South = one preset at a time.
# Paper configuration: blind-analyst (isotropic) covariance estimated from an INDEPENDENT
# set of sims (--indep_cov), so every null distribution is out-of-sample.
#
# Every run writes its full log next to its npz:  results/fiducial_<S>/asym_..._covisotropic_indepcov.log
# This master log keeps start / end / return code. Output printed before a run opens its own
# log goes to results/logs/launch_<S>_<cov>.txt (deleted when the run succeeds).
#
# Cost: a South that is not cached = 1000 new sims (~1.5 h with 80 threads). The first run
# also generates the 1000 independent covariance sims (once per mask). Cached Souths only refit.
#
#   chmod +x scripts/run_PRESETscan.sh
#   mkdir -p results/logs
#   nohup scripts/run_PRESETscan.sh > results/logs/scan_all_master.log 2>&1 &

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hemicosmo
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

PY="python"
NORTH="fiducial"

# --- Souths, grouped by injected parameter (comment out what you do not want) -----------
# Values ~ in 1-sky sigma of the H6+V6 mask: sigma(H0)=0.86, sigma(ombh2)=1.8e-4,
# sigma(omch2)=2.0e-3, sigma(ns)=4.9e-3, sigma(As e^-2tau)=9.4e-3
# Quasi-null presets (1206omc = fiducial exactly, 221omb ~ -0.1 sigma) are left out.
SOUTHS=(
    62H0 65H0 68.5H0 71H0 74H0                 # -5.7 -2.2 +1.9 +4.8 +8.3 sigma (cached: refit only)
    092ns 094ns 096ns 098ns 100ns              # -8.7 -4.6 -0.5 +3.6 +7.6 sigma
    215omb 218omb 224omb 227omb                # -3.5 -1.8 +1.6 +3.3 sigma
    1110omc 1150omc 1178omc 1234omc 1262omc 1302omc   # -4.9 -2.9 -1.4 +1.4 +2.9 +4.9 sigma
    200As 204As 208As 212As 216As              # -8.9 -5.0 -1.2 +2.9 +7.0 sigma
)
COV_MODES=(isotropic)          # add 'stitched' for the decomposition runs
INDEP="--indep_cov"            # "" to reproduce the in-sample (old) behaviour
EXTRA="--expected"             # add --compare_cov once if you want the covariance comparison again

COMMON="--north ${NORTH} --nside 1024 --delta_l 30 --lmin 32 --apod 1. \
        --blend 3. --beam 0.0 --nsims 1000 --n_threads 80 \
        --phase_mode independent --minuit \
        --naive_mask_h 6 --naive_mask_v 6"

# Reuse v1 sim caches (same seeds -> identical sims; values NOT verifiable: only if the presets
# were not edited since). Renamed presets (68.5H0 <- 68H0, 200As <- 2As) are mapped automatically.
ADOPT="--adopt_legacy_cache"   # or "" to regenerate

LOGDIR="results/logs"
mkdir -p "$LOGDIR"
echo "=== scan started $(date '+%F %T')  N=${NORTH}  ${#SOUTHS[@]} Souths  cov=(${COV_MODES[*]})  ${INDEP} ==="

n_fail=0
for CM in "${COV_MODES[@]}"; do
    for S in "${SOUTHS[@]}"; do
        LAUNCH="${LOGDIR}/launch_${S}_${CM}.txt"
        t0=$(date +%s)
        echo "--- [$(date +%T)] S=${S} cov=${CM}"
        $PY scripts/run_asymmetry.py --south "$S" $COMMON --cov_mode "$CM" $INDEP $EXTRA $ADOPT \
            > "$LAUNCH" 2>&1
        rc=$?
        dt=$(( $(date +%s) - t0 ))
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
echo "=== scan finished $(date '+%F %T')  failures: ${n_fail} ==="
echo "next:  python scripts/plot_total.py --mask maskH6_maskV6l00 --cov_mode isotropic --indep_cov"
