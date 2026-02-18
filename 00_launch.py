"""Detached launcher for 00_run_full_label.py on Windows."""
import subprocess, sys, os

base = os.path.dirname(os.path.abspath(__file__))
log  = os.path.join(base, "00_run_full_label_log.txt")

proc = subprocess.Popen(
    [sys.executable, "-X", "utf8", "00_run_full_label.py"],
    stdout=open(log, "a", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    cwd=base,
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
)
print(f"Launched PID {proc.pid} -> {log}")
