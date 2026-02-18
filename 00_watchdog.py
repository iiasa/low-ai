"""
00_watchdog.py — restart 00_run_full_label.py if it crashes.
Runs in foreground; kill with Ctrl+C.
"""
import subprocess, time, os, sys
from datetime import datetime

SCRIPT = "00_run_full_label.py"
LOG    = "00_run_full_label_log.txt"
CHECK_INTERVAL = 30   # seconds between alive checks

def stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

proc = None

def launch():
    global proc
    print(f"[{stamp()}] Starting {SCRIPT} ...")
    with open(LOG, "a", encoding="utf-8") as lf:
        lf.write(f"\n\n[WATCHDOG] Restarted at {stamp()}\n\n")
    proc = subprocess.Popen(
        [sys.executable, SCRIPT],
        stdout=open(LOG, "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    print(f"[{stamp()}] PID {proc.pid}")
    return proc

launch()

try:
    while True:
        time.sleep(CHECK_INTERVAL)
        if proc.poll() is not None:
            rc = proc.returncode
            if rc == 0:
                print(f"[{stamp()}] Process finished cleanly (rc=0). Done.")
                break
            else:
                print(f"[{stamp()}] Process exited with rc={rc}. Restarting in 10s...")
                time.sleep(10)
                launch()
        else:
            # Still running — check tracker for signs of life
            trk = "00_tracker_full.txt"
            if os.path.exists(trk):
                mtime = os.path.getmtime(trk)
                age   = time.time() - mtime
                # If tracker hasn't been updated in >30 min, something may be stuck
                if age > 1800:
                    print(f"[{stamp()}] Tracker not updated for {age/60:.0f}min — process may be stalled.")
except KeyboardInterrupt:
    print(f"\n[{stamp()}] Watchdog stopped by user.")
    if proc and proc.poll() is None:
        proc.terminate()
