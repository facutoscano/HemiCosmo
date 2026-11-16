#!/usr/bin/env bash
#
# Causal test of the second-order mechanism: run the simulation pipeline on the
# theta_*-preserving / theta_*-matched companions of the sources in the config
#   <src>_thfix    Var_w(ln theta_*) = 0     (only the non-dilation part survives)
#   <src>_thmatch  same Var_w as <src>       (pure dilation)
# The sources themselves were already run by run_PRESETscan.sh.
#
# Everything (geometry, nsims, covariance, fit) comes from the single config file.
# Before the first run:   python scripts/make_theta_presets.py --config configs/paper.toml
# Cost: 1000 new sims per South (~1.5 h with 80 threads). Predict first, for free:
#   python scripts/run_prediction.py --config configs/paper.toml --scan \
#       --souths 1110omc 1110omc_thfix 1110omc_thmatch
#
#   chmod +x scripts/run_injections.sh
#   nohup scripts/run_injections.sh > results/logs/injections_master.log 2>&1 &

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hemicosmo
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

CONFIG="configs/paper.toml"
SOUTHS=(
    1110omc_thfix 1110omc_thmatch
    1302omc_thfix 1302omc_thmatch
    # 1150omc_thfix 1150omc_thmatch 1262omc_thfix 1262omc_thmatch
)

LOGDIR="results/logs"
mkdir -p "$LOGDIR"
echo "=== injections started $(date '+%F %T')  config=${CONFIG}  ${#SOUTHS[@]} Souths ==="
n_fail=0
for S in "${SOUTHS[@]}"; do
    LAUNCH="${LOGDIR}/launch_${S}.txt"
    t0=$(date +%s)
    echo "--- [$(date +%T)] S=${S}"
    python scripts/run_asymmetry.py --config "$CONFIG" --south "$S" > "$LAUNCH" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -ne 0 ]; then
        n_fail=$((n_fail + 1))
        echo "    !! FAIL rc=${rc} after ${dt}s  launch output kept: ${LAUNCH}"
    else
        echo "    ok after ${dt}s"
        rm -f "$LAUNCH"
    fi
done
echo "=== injections finished $(date '+%F %T')  failures: ${n_fail} ==="
