"""
Logging utility:
-tee_output(path): everything written to the terminal (Python prints, warnings,
 tracebacks, AND C/Fortran-level output from CAMB / NaMaster / healpy, AND the
 output of worker processes) is also written to `path`.

Implemented at the file-descriptor level (fd 1 and 2 are redirected into a
`tee` process), so nothing bypasses it. Falls back to a Python-level tee if the
`tee` binary is not available (then C-level output is not captured).
"""

from __future__ import annotations
import os
import sys
import shutil
import datetime
import platform
import subprocess
import traceback
from contextlib import contextmanager


def _header(path):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (f"# log: {os.path.basename(path)}\n"
            f"# started: {now}   host: {platform.node()}   pid: {os.getpid()}\n"
            f"# cwd: {os.getcwd()}\n"
            f"# command: {' '.join([sys.executable] + sys.argv)}\n"
            + "#" * 80 + "\n")


class _PyTee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def tee_output(path, mode: str = "w"):
    """
    Duplicate stdout+stderr into `path` for the duration of the block.
    mode='w' overwrites (same policy as the .npz it accompanies); 'a' appends.
    An exception inside the block is printed (so the traceback is in the log)
    and turned into SystemExit(1).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, mode) as fh:
        fh.write(_header(path))
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(line_buffering=True)
        except Exception:
            pass
    sys.stdout.flush(); sys.stderr.flush()

    use_fd = shutil.which("tee") is not None
    if use_fd:
        tee = subprocess.Popen(["tee", "-a", path], stdin=subprocess.PIPE)
        saved_out, saved_err = os.dup(1), os.dup(2)
        os.dup2(tee.stdin.fileno(), 1)
        os.dup2(tee.stdin.fileno(), 2)
    else:
        fh = open(path, "a", buffering=1)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _PyTee(old_out, fh), _PyTee(old_err, fh)

    failed = False
    t0 = datetime.datetime.now()
    try:
        yield path
    except KeyboardInterrupt:
        print("\n[log] interrupted by user", flush=True)
        failed = True
        raise
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        failed = True
    finally:
        dt = datetime.datetime.now() - t0
        print(f"\n[log] {'FAILED' if failed else 'finished'} after {dt} -> {path}", flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        if use_fd:
            os.dup2(saved_out, 1); os.dup2(saved_err, 2)
            os.close(saved_out); os.close(saved_err)
            tee.stdin.close()
            tee.wait()
        else:
            sys.stdout, sys.stderr = old_out, old_err
            fh.close()
    if failed:
        raise SystemExit(1)
