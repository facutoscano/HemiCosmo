#!/usr/bin/env python
"""
Ordena results/: separa lo producido por el código actual (v4) de los tests viejos.

Por defecto NO toca nada (dry run): imprime la clasificación. Con --apply MUEVE lo viejo a
<proyecto>/results_archive/<fecha>/ (fuera de results/, para que plot_scan/plot_total no lo
vuelvan a leer), preservando subcarpetas, y deja un manifest.csv. No borra nada: cuando
revises el archivo, lo borrás vos.

Criterios
  asym_*.npz           actual  <=> tiene la clave 'fit_errors_fisher' (iteración 3 = v4)
  asym_*.log           actual  <=> su npz es actual; sin npz: FAILED/interrumpido -> viejo,
                                   sin pie y modificado hace < --running_h horas -> se deja (¿corriendo?)
  figuras de una corrida actual <=> mtime >= '# started:' del log de esa corrida (si es más
                                   vieja, es de una versión anterior y la corrida nueva no la rehízo)
  validation/, solve_region/   <=> mtime dentro de la ventana [started, fin] de algún log de la carpeta
                                   (los logs existen desde la iteración 2; la validación no cambió desde ahí)
  resúmenes en la raíz (plot_scan / plot_total) <=> más nuevos que el npz actual más reciente;
                                   si no, quedan desactualizados -> se archivan y se regeneran
  Maps_*.png                   se dejan (no dependen de la versión)
  cualquier otra cosa          'unknown': se lista y se deja, salvo --move_unknown

  --final_only  archiva además las corridas v4 que no son _covisotropic_indepcov
                (covarianza in-sample o stitched: superadas por el fix (a)).

Uso
  python scripts/tidy_results.py                    # ver qué haría
  python scripts/tidy_results.py --final_only       # ídem, quedándose sólo con isotropic+indepcov
  python scripts/tidy_results.py --apply [--final_only]
  python scripts/tidy_results.py --index_only       # sólo regenerar results/INDEX.md
Después de --apply: regenerar los resúmenes (plot_scan.py / plot_total.py) con las corridas que quedaron.
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
SLACK = 300.0                                   # s de tolerancia en las ventanas de tiempo
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
            continue                                    # raíz: después (necesita newest_current_npz)

        # ---------------- carpetas con log por ventana de tiempo
        if sub in ("validation", "solve_region") or sub.startswith("solve_region"):
            logs = [read_log(p) | dict(path=p) for p in files if p.endswith(".log")]
            for p in files:
                t = os.path.getmtime(p)
                win = [L for L in logs if L["started"] and L["started"] - SLACK <= t <= L["end"] + SLACK]
                if p.endswith(".log"):
                    L = read_log(p)
                    if L["status"] == "open" and now - L["end"] < running_h * 3600:
                        add(p, "keep", sub, "log abierto y reciente (¿corriendo?)")
                    elif L["status"] == "failed":
                        add(p, "archive", sub, "corrida FAILED/interrumpida")
                    else:
                        add(p, "keep", sub, f"log {L['status']} {fmt_t(L['started'])}")
                elif win:
                    add(p, "keep", sub, f"producido por la corrida del {fmt_t(win[0]['started'])}")
                else:
                    add(p, "archive", sub, "sin log que lo cubra (anterior a la iteración 2)")
            continue

        # ---------------- carpetas de corridas asym (results/<tag>/)
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
                add(p, "archive", "run", "npz anterior a v4 (sin fit_errors_fisher)")
            elif superseded:
                add(p, "archive", "run", f"v4 pero '{suffix or '_stitched'}' (superada por isotropic+indepcov)")
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
                        "log de corrida v4" if S["current"] else "log de corrida archivada")
                else:
                    L = read_log(p)
                    if L["status"] == "open" and now - L["end"] < running_h * 3600:
                        add(p, "keep", "run", "log sin npz, abierto y reciente (¿corriendo?)")
                    else:
                        add(p, "archive", "run", f"log sin npz ({L['status']})")
                continue
            if b.startswith("asym_") and b.endswith(FIG_EXT):
                base = os.path.splitext(b)[0]
                owner = None
                for stem in stems:
                    if base != stem and base.endswith("_" + stem[len("asym_"):]):
                        owner = stem
                        break
                if owner is None:
                    add(p, "archive", "figure", "figura huérfana (su npz no existe)")
                    continue
                S = stems[owner]
                if not S["current"]:
                    add(p, "archive", "figure", "figura de corrida archivada")
                    continue
                t0 = S["log"]["started"] if S["log"] and S["log"]["started"] else os.path.getmtime(S["npz"]) - 6 * 3600
                if t >= t0 - SLACK:
                    add(p, "keep", "figure", "figura de corrida v4")
                    for r in runs:
                        if r["stem"] == owner:
                            r["figures"].append(b)
                else:
                    add(p, "archive", "figure", f"más vieja que la corrida ({fmt_t(t)} < {fmt_t(t0)}): versión anterior")
                continue
            if b.startswith("Maps_"):
                add(p, "keep", "maps", "mapas (independientes de la versión)")
                continue
            add(p, "archive" if move_unknown else "keep", "unknown", "no reconocido")

    # ---------------- raíz
    for p in by_dir.get(os.path.normpath(results_dir), []):
        b = os.path.basename(p)
        if SUMMARY_RE.match(b):
            t = os.path.getmtime(p)
            if final_only and "isotropic_indepcov" not in b:
                add(p, "archive", "summary", "resumen de una covarianza no final (no isotropic_indepcov)")
            elif newest_current_npz and t >= newest_current_npz - SLACK:
                add(p, "keep", "summary", "resumen posterior a la última corrida v4")
            else:
                add(p, "archive", "summary", "resumen desactualizado: regenerar con plot_scan/plot_total")
        elif b.startswith("Maps_"):
            add(p, "keep", "maps", "mapas")
        else:
            add(p, "archive" if move_unknown else "keep", "unknown", "no reconocido (raíz)")
    return rows, runs


# ----------------------------------------------------------------------------- outputs
def report(rows):
    cnt = Counter((r["action"], r["category"]) for r in rows)
    size = defaultdict(int)
    for r in rows:
        size[r["action"]] += r["size"]
    print("\n=== resumen ===")
    for (a, c), n in sorted(cnt.items()):
        print(f"  {a:8s} {c:10s} {n:5d}")
    print(f"  keep: {size['keep']/1e9:.2f} GB   archive: {size['archive']/1e9:.2f} GB")
    print("\n=== a archivar (por motivo) ===")
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
        print(f"\n=== no reconocidos ({len(unk)}; se {'archivan' if any(r['action']=='archive' for r in rows if r['category']=='unknown') else 'dejan'}) ===")
        for x in unk[:20]:
            print(f"      {x}")


def apply(rows, results_dir, archive_root):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    dest_root = os.path.join(archive_root, stamp)
    todo = [r for r in rows if r["action"] == "archive"]
    if not todo:
        print("\nnada para archivar")
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
    # carpetas vacías
    for d in sorted(glob.glob(os.path.join(results_dir, "**", ""), recursive=True), key=len, reverse=True):
        if os.path.normpath(d) != os.path.normpath(results_dir) and not os.listdir(d):
            os.rmdir(d)
    print(f"\n{len(todo)} archivos movidos a {dest_root}  (manifest.csv adentro)")
    return dest_root


def write_index(results_dir, runs, rows):
    lines = ["# results/ — corridas vigentes (código v4)", "",
             f"_generado {fmt_t(datetime.datetime.now().timestamp())} por scripts/tidy_results.py_", "",
             "| tag | máscara | cov | indep | fecha | nsims | " + " | ".join(f"Δ{n}/σ" for n in PARAM_NAMES) + " |",
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
        lines.append(f"| {r['tag']} | {r['mask']} | {r['cov']} | {'sí' if r['indep'] else 'no'} | {date} | {nsims} | "
                     + " | ".join(dev) + " |")
    lines += ["", "Δp/σ = (media de los ajustes por cielo − predicción de primer orden) / σ por cielo.", "",
              "## Figuras por corrida", ""]
    for r in sorted(runs, key=lambda r: (r["tag"], r["cov"], r["indep"])):
        lines.append(f"- **{r['tag']}** ({r['mask']}, {r['cov']}{', indepcov' if r['indep'] else ''})")
        for f in sorted(r["figures"]):
            lines.append(f"  - `{r['tag']}/{f}`")
    other = [r["rel"] for r in rows if r["action"] == "keep" and r["category"] in ("validation", "summary", "maps")]
    if other:
        lines += ["", "## Validación, resúmenes y mapas", ""] + [f"- `{x}`" for x in sorted(other)
                                                              if not x.endswith((".npz", ".log"))]
    path = os.path.join(results_dir, "INDEX.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"índice -> {path}")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=RESULTS_DIR)
    p.add_argument("--archive_dir", default=os.path.join(PROJECT_DIR, "results_archive"))
    p.add_argument("--apply", action="store_true", help="mover de verdad (sin esto: dry run)")
    p.add_argument("--final_only", action="store_true",
                   help="dejar sólo corridas v4 _covisotropic_indepcov")
    p.add_argument("--move_unknown", action="store_true", help="archivar también lo no reconocido")
    p.add_argument("--running_h", type=float, default=3.0,
                   help="un log sin pie modificado hace menos de esto se considera en curso")
    p.add_argument("--index_only", action="store_true")
    p.add_argument("--force", action="store_true", help="aplicar aunque haya corridas activas")
    a = p.parse_args()

    if os.path.commonpath([os.path.abspath(a.archive_dir), os.path.abspath(a.results_dir)]) == \
            os.path.abspath(a.results_dir):
        raise SystemExit("--archive_dir no puede estar dentro de results/ (load_runs lo volvería a leer)")

    rows, runs = classify(a.results_dir, a.final_only, a.move_unknown, a.running_h)
    if a.index_only:
        write_index(a.results_dir, runs, rows)
        return
    report(rows)
    if not a.apply:
        print("\n(dry run: no se movió nada; repetir con --apply)")
        return
    busy = jobs_running()
    if busy and not a.force:
        print("\nHay corridas activas:\n  " + "\n  ".join(busy))
        raise SystemExit("esperá a que terminen o usá --force")
    apply(rows, a.results_dir, a.archive_dir)
    rows, runs = classify(a.results_dir, a.final_only, a.move_unknown, a.running_h)
    write_index(a.results_dir, runs, rows)


if __name__ == "__main__":
    main()
