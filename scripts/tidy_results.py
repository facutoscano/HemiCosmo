#!/usr/bin/env python
"""
Tidy up results/: separate what the current code (v4) produced from old tests.

By default it touches NOTHING (dry run): it prints the classification. With --apply it MOVES
the old files to <project>/results_archive/<date>/ (outside results/, so that plot_scan /
plot_total never read them again), keeping sub-folders, and writes a manifest.csv. Nothing is
deleted: once you have checked the archive, delete it yourself.

Criteria
  asym_*.npz           current <=> has the key 'fit_errors_fisher' (iteration 3 = v4)
  asym_*.log           current <=> its npz is current; without npz: FAILED/interrupted -> old,
                                   no footer and modified < --running_h hours ago -> kept (running?)
  figures of a current run     <=> mtime >= '# started:' of that run's log (if older, it comes
                                   from a previous version and the new run did not redo it)
  validation/, solve_region/   <=> mtime inside the [started, end] window of some log of the folder
                                   (logs exist since iteration 2; validation did not change since)
  summaries in the root (plot_scan / plot_total) <=> newer than the most recent current npz;
                                   otherwise they are outdated -> archived, to be regenerated
  Maps_*.png                   kept (they do not depend on the version)
  prediction/, observational/  kept (new in iteration 5)
  anything else                'unknown': listed and kept, unless --move_unknown

  --final_only  also archives the v4 runs that are not _covisotropic_indepcov
                (in-sample or stitched covariance: superseded by fix (a)).

Usage
  python scripts/tidy_results.py                    # see what it would do
  python scripts/tidy_results.py --final_only       # same, keeping only isotropic+indepcov
  python scripts/tidy_results.py --apply [--final_only]
  python scripts/tidy_results.py --index_only       # only regenerate results/INDEX.md
After --apply: regenerate the summaries (plot_scan.py / plot_total.py) with the runs left.
"""
import os
import re
import sys
import csv
import glob
import shutil
import argparse
import datetime
import subprocess
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import RESULTS_DIR, PROJECT_DIR, PARAM_NAMES
from hemcosmo.scanio import parse_name

V4_KEY = "fit_errors_fisher"
FINAL_SUFFIX = "_covisotropic_indepcov"
FIG_EXT = (".pdf", ".png", ".svg", ".jpg")
SLACK = 300.0                                   # s of tolerance in the time windows
SUMMARY_RE = re.compile(r"^(\w+scan_(response|secondorder).*|total_secondorder_.*)\.(png|pdf|npz)$")
_STARTED = re.compile(r"^# started: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


# ----------------------------------------------------------------------------- helpers
def read_log(path):
    """-> dict(started=epoch|None, status='finished'|'failed'|'open', end=mtime, command=str)"""
    started, command, status = None, "", "open"
    try:
        with open(path, errors="ignore") as fh:
            head = [next(fh, "") for _ in range(6)]
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 4000))
            tail = fh.read()
    except OSError:
        return dict(started=None, status="open", end=os.path.getmtime(path), command="")
    for line in head:
        m = _STARTED.match(line)
        if m:
            started = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        if line.startswith("# command:"):
            command = line[len("# command:"):].strip()
    if "[log] finished after" in tail:
        status = "finished"
    elif "[log] FAILED after" in tail or "[log] interrupted" in tail:
        status = "failed"
    return dict(started=started, status=status, end=os.path.getmtime(path), command=command)


def npz_keys(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            return set(d.files)
    except Exception:
        return None


def run_suffix_of(stem):
    for s in ("_covisotropic_indepcov", "_covstitched_indepcov", "_indepcov",
              "_covisotropic", "_covstitched"):
        if stem.endswith(s):
            return s
    return ""


def fmt_t(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else "?"


def jobs_running():
    try:
        out = subprocess.run(["pgrep", "-af", r"run_asymmetry|run_validation|solve_region"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    return [l for l in out.splitlines() if "tidy_results" not in l and l.strip()]


# ----------------------------------------------------------------------------- classification
def classify(results_dir, final_only=False, move_unknown=False, running_h=3.0):
    """
    -> list of dict(path, rel, action in {keep, archive}, category, reason) + list of current runs
    """
    now = datetime.datetime.now().timestamp()
    rows, runs = [], []

    def add(path, action, category, reason):
        rows.append(dict(path=path, rel=os.path.relpath(path, results_dir), action=action,
                         category=category, reason=reason, size=os.path.getsize(path),
                         mtime=os.path.getmtime(path)))

    all_files = sorted(p for p in glob.glob(os.path.join(results_dir, "**", "*"), recursive=True)
                       if os.path.isfile(p) and os.path.basename(p) != "INDEX.md")
    by_dir = defaultdict(list)
    for p in all_files:
        by_dir[os.path.dirname(p)].append(p)

    newest_current_npz = 0.0
    for d, files in by_dir.items():
        sub = os.path.relpath(d, results_dir)
        names = {os.path.basename(p): p for p in files}
        done = set()

        if sub == ".":
            continue                                    # root: later (needs newest_current_npz)

        # ---------------- outputs of the prediction / observational scripts (iteration 5)
        top = sub.split(os.sep)[0]
        if top in ("prediction", "observational", "logs"):
            for p in files:
                add(p, "keep", top, f"{top} output (not versioned by this script)")
            continue

        # ---------------- folders with a log, by time window
        if sub in ("validation", "solve_region") or sub.startswith("solve_region"):
            logs = [read_log(p) | dict(path=p) for p in files if p.endswith(".log")]
            for p in files:
                t = os.path.getmtime(p)
                win = [L for L in logs if L["started"] and L["started"] - SLACK <= t <= L["end"] + SLACK]
                if p.endswith(".log"):
                    L = read_log(p)
                    if L["status"] == "open" and now - L["end"] < running_h * 3600:
                        add(p, "keep", sub, "open and recent log (running?)")
                    elif L["status"] == "failed":
                        add(p, "archive", sub, "FAILED/interrupted run")
                    else:
                        add(p, "keep", sub, f"log {L['status']} {fmt_t(L['started'])}")
                elif win:
                    add(p, "keep", sub, f"produced by the run of {fmt_t(win[0]['started'])}")
                else:
                    add(p, "archive", sub, "no log covers it (before iteration 2)")
            continue

        # ---------------- asym run folders (results/<tag>/)
        npzs = [p for p in files if os.path.basename(p).startswith("asym_") and p.endswith(".npz")]
        stems = {}
        for p in npzs:
            stem = os.path.basename(p)[:-4]
            keys = npz_keys(p)
            log = os.path.join(d, stem + ".log")
            L = read_log(log) if os.path.exists(log) else None
            current = keys is not None and V4_KEY in keys
            suffix = run_suffix_of(stem)
            superseded = current and final_only and suffix != FINAL_SUFFIX
            stems[stem] = dict(current=current and not superseded, log=L, npz=p, suffix=suffix)
            if keys is None:
                add(p, "archive", "run", "npz ilegible/corrupto")
            elif not current:
                add(p, "archive", "run", "npz older than v4 (no fit_errors_fisher)")
            elif superseded:
                add(p, "archive", "run", f"v4 but '{suffix or '_stitched'}' (superseded by isotropic+indepcov)")
            else:
                add(p, "keep", "run", "v4")
                newest_current_npz = max(newest_current_npz, os.path.getmtime(p))
                tag = sub
                mtag, phase, cov, indep = parse_name(p)
                runs.append(dict(tag=tag, stem=stem, npz=p, mask=mtag, phase=phase, cov=cov,
                                 indep=indep, log=L, figures=[]))
            done.add(p)

        for p in files:
            if p in done:
                continue
            b = os.path.basename(p)
            t = os.path.getmtime(p)
            if b.endswith(".log") and b.startswith("asym_"):
                stem = b[:-4]
                if stem in stems:
                    S = stems[stem]
                    add(p, "keep" if S["current"] else "archive", "run",
                        "log of a v4 run" if S["current"] else "log of an archived run")
                else:
                    L = read_log(p)
                    if L["status"] == "open" and now - L["end"] < running_h * 3600:
                        add(p, "keep", "run", "log without npz, open and recent (running?)")
                    else:
                        add(p, "archive", "run", f"log without npz ({L['status']})")
                continue
            if b.startswith("asym_") and b.endswith(FIG_EXT):
                base = os.path.splitext(b)[0]
                owner = None
                for stem in stems:
                    if base != stem and base.endswith("_" + stem[len("asym_"):]):
                        owner = stem
                        break
                if owner is None:
                    add(p, "archive", "figure", "orphan figure (its npz does not exist)")
                    continue
                S = stems[owner]
                if not S["current"]:
                    add(p, "archive", "figure", "figure of an archived run")
                    continue
                t0 = S["log"]["started"] if S["log"] and S["log"]["started"] else os.path.getmtime(S["npz"]) - 6 * 3600
                if t >= t0 - SLACK:
                    add(p, "keep", "figure", "figure of a v4 run")
                    for r in runs:
                        if r["stem"] == owner:
                            r["figures"].append(b)
                else:
                    add(p, "archive", "figure", f"older than the run ({fmt_t(t)} < {fmt_t(t0)}): previous version")
                continue
            if b.startswith("Maps_"):
                add(p, "keep", "maps", "maps (independent of the version)")
                continue
            add(p, "archive" if move_unknown else "keep", "unknown", "no reconocido")

    # ---------------- root
    for p in by_dir.get(os.path.normpath(results_dir), []):
        b = os.path.basename(p)
        if SUMMARY_RE.match(b):
            t = os.path.getmtime(p)
            if final_only and "isotropic_indepcov" not in b:
                add(p, "archive", "summary", "summary of a non-final covariance (not isotropic_indepcov)")
            elif newest_current_npz and t >= newest_current_npz - SLACK:
                add(p, "keep", "summary", "summary newer than the last v4 run")
            else:
                add(p, "archive", "summary", "outdated summary: regenerate with plot_scan/plot_total")
        elif b.startswith("Maps_"):
            add(p, "keep", "maps", "mapas")
        else:
            add(p, "archive" if move_unknown else "keep", "unknown", "not recognised (root)")
    return rows, runs


# ----------------------------------------------------------------------------- outputs
def report(rows):
    cnt = Counter((r["action"], r["category"]) for r in rows)
    size = defaultdict(int)
    for r in rows:
        size[r["action"]] += r["size"]
    print("\n=== summary ===")
    for (a, c), n in sorted(cnt.items()):
        print(f"  {a:8s} {c:10s} {n:5d}")
    print(f"  keep: {size['keep']/1e9:.2f} GB   archive: {size['archive']/1e9:.2f} GB")
    print("\n=== to archive (by reason) ===")
    by_reason = defaultdict(list)
    for r in rows:
        if r["action"] == "archive":
            by_reason[r["reason"]].append(r["rel"])
    for reason, lst in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  [{len(lst)}] {reason}")
        for x in lst[:8]:
            print(f"      {x}")
        if len(lst) > 8:
            print(f"      ... (+{len(lst) - 8})")
    unk = [r["rel"] for r in rows if r["category"] == "unknown"]
    if unk:
        print(f"\n=== not recognised ({len(unk)}; {'archived' if any(r['action']=='archive' for r in rows if r['category']=='unknown') else 'kept'}) ===")
        for x in unk[:20]:
            print(f"      {x}")


def apply(rows, results_dir, archive_root):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    dest_root = os.path.join(archive_root, stamp)
    todo = [r for r in rows if r["action"] == "archive"]
    if not todo:
        print("\nnothing to archive")
        return None
    os.makedirs(dest_root, exist_ok=True)
    with open(os.path.join(dest_root, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["original", "category", "reason", "bytes", "mtime"])
        for r in todo:
            dst = os.path.join(dest_root, r["rel"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(r["path"], dst)
            w.writerow([r["rel"], r["category"], r["reason"], r["size"], fmt_t(r["mtime"])])
    # empty folders
    for d in sorted(glob.glob(os.path.join(results_dir, "**", ""), recursive=True), key=len, reverse=True):
        if os.path.normpath(d) != os.path.normpath(results_dir) and not os.listdir(d):
            os.rmdir(d)
    print(f"\n{len(todo)} files moved to {dest_root}  (manifest.csv inside)")
    return dest_root


def write_index(results_dir, runs, rows):
    lines = ["# results/ — current runs (code v4)", "",
             f"_generated {fmt_t(datetime.datetime.now().timestamp())} by scripts/tidy_results.py_", "",
             "| tag | mask | cov | indep | date | nsims | " + " | ".join(f"Δ{n}/σ" for n in PARAM_NAMES) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(PARAM_NAMES)]
    for r in sorted(runs, key=lambda r: (r["mask"], r["cov"], r["indep"], r["tag"])):
        with np.load(r["npz"], allow_pickle=False) as d:
            nsims = int(d["nsims"]) if "nsims" in d.files else -1
            dev = ["–"] * len(PARAM_NAMES)
            if {"fits_asym", "pred_first_order_weights", "sigma_null"} <= set(d.files):
                fa = np.asarray(d["fits_asym"], float)
                pred = np.asarray(d["pred_first_order_weights"], float)
                sig = np.asarray(d["sigma_null"], float)
                if fa.ndim == 2 and fa.shape[1] >= len(PARAM_NAMES):
                    z = (fa.mean(0)[:len(PARAM_NAMES)] - pred[:len(PARAM_NAMES)]) / sig[:len(PARAM_NAMES)]
                    dev = [f"{v:+.2f}" for v in z]
        date = fmt_t(r["log"]["started"]) if r["log"] else "?"
        lines.append(f"| {r['tag']} | {r['mask']} | {r['cov']} | {'yes' if r['indep'] else 'no'} | {date} | {nsims} | "
                     + " | ".join(dev) + " |")
    lines += ["", "Δp/σ = (mean of the per-sky fits − first-order prediction) / 1-sky σ.", "",
              "## Figures per run", ""]
    for r in sorted(runs, key=lambda r: (r["tag"], r["cov"], r["indep"])):
        lines.append(f"- **{r['tag']}** ({r['mask']}, {r['cov']}{', indepcov' if r['indep'] else ''})")
        for f in sorted(r["figures"]):
            lines.append(f"  - `{r['tag']}/{f}`")
    other = [r["rel"] for r in rows if r["action"] == "keep" and r["category"] in ("validation", "summary", "maps")]
    if other:
        lines += ["", "## Validation, summaries and maps", ""] + [f"- `{x}`" for x in sorted(other)
                                                              if not x.endswith((".npz", ".log"))]
    path = os.path.join(results_dir, "INDEX.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"index -> {path}")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=RESULTS_DIR)
    p.add_argument("--archive_dir", default=os.path.join(PROJECT_DIR, "results_archive"))
    p.add_argument("--apply", action="store_true", help="really move the files (without it: dry run)")
    p.add_argument("--final_only", action="store_true",
                   help="keep only v4 _covisotropic_indepcov runs")
    p.add_argument("--move_unknown", action="store_true", help="also archive what is not recognised")
    p.add_argument("--running_h", type=float, default=3.0,
                   help="a log without footer modified less than this ago is considered running")
    p.add_argument("--index_only", action="store_true")
    p.add_argument("--force", action="store_true", help="apply even if there are active runs")
    a = p.parse_args()

    if os.path.commonpath([os.path.abspath(a.archive_dir), os.path.abspath(a.results_dir)]) == \
            os.path.abspath(a.results_dir):
        raise SystemExit("--archive_dir cannot be inside results/ (load_runs would read it again)")

    rows, runs = classify(a.results_dir, a.final_only, a.move_unknown, a.running_h)
    if a.index_only:
        write_index(a.results_dir, runs, rows)
        return
    report(rows)
    if not a.apply:
        print("\n(dry run: nothing was moved; repeat with --apply)")
        return
    busy = jobs_running()
    if busy and not a.force:
        print("\nThere are active runs:\n  " + "\n  ".join(busy))
        raise SystemExit("wait until they finish or use --force")
    apply(rows, a.results_dir, a.archive_dir)
    rows, runs = classify(a.results_dir, a.final_only, a.move_unknown, a.running_h)
    write_index(a.results_dir, runs, rows)


if __name__ == "__main__":
    main()
