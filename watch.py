#!/usr/bin/env python3
"""Refresh the public dashboard every 15 minutes; single-instance local worker."""
import argparse
import datetime as dt
import fcntl
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--checkout', type=Path, required=True)
a = p.parse_args()
with (a.checkout / '.git/dashboard-refresh.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        print(dt.datetime.now(dt.timezone.utc).isoformat(), 'export check', flush=True)
        try:
            subprocess.run([sys.executable, str(a.repo/'dashboard/publish.py'), '--repo',str(a.repo),'--checkout',str(a.checkout)], check=True, timeout=180)
        except (subprocess.SubprocessError, OSError) as error:
            print(type(error).__name__, 'refresh failed; retry in 15 minutes', flush=True)
        time.sleep(900)
