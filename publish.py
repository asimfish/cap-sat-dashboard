#!/usr/bin/env python3
"""Publish the allowlisted dashboard files, never the experiment repository."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys

def run(args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--checkout', type=Path, required=True)
    a = p.parse_args()
    source = a.repo.resolve() / 'dashboard'
    dest = a.checkout.resolve()
    if dest == a.repo.resolve() or not (dest / '.git').is_dir():
        raise SystemExit('Expected a separate dashboard git checkout')
    remote = run(['git','remote','get-url','origin'], dest)
    if remote not in ['https://github.com/asimfish/cap-sat-dashboard.git','git@github.com:asimfish/cap-sat-dashboard.git']:
        raise SystemExit('Unexpected publishing remote')
    run(['git','pull','--ff-only'], dest)
    run([sys.executable, str(source/'build.py'), '--repo',str(a.repo),'--out',str(source/'site')])
    previous = dest/'site/status.json'
    current = json.loads((source/'site/status.json').read_text())
    if previous.exists():
        old = json.loads(previous.read_text())
        age = (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(old['generated'])).total_seconds()
        unchanged = old['source_sha256'] == current['source_sha256'] and old['issues'] == current['issues'] and old.get('native') == current.get('native') and old.get('comparison') == current.get('comparison') and old.get('performance_plan') == current.get('performance_plan') and old.get('pilot') == current.get('pilot') and old.get('hybrid') == current.get('hybrid')
        same_code = all((dest/n).exists() and (dest/n).read_bytes() == (source/n).read_bytes() for n in ['build.py','template.html','issues.json','performance_plan.json','README.md','publish.py','watch.py','test_build.py','EVIDENCE.md','site/evidence.html','LICENSE','.github/workflows/pages.yml'])
        if unchanged and same_code and age < 3600:
            print('No change; next freshness export within one hour')
            return
    names = ['build.py','template.html','issues.json','performance_plan.json','README.md','publish.py','watch.py','test_build.py','LICENSE','site/index.html','site/status.json','site/evidence.html','EVIDENCE.md','.github/workflows/pages.yml']
    for name in names:
        src = source/name
        if src.exists():
            (dest/name).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest/name)
    run(['git','add','--']+[n for n in names if (dest/n).exists()],dest)
    if run(['git','diff','--cached','--name-only'],dest):
        run(['git','commit','-m','Refresh audited CAP-SAT progress snapshot'],dest)
        run(['git','push','origin','main'],dest)
    print('Dashboard published')

if __name__ == '__main__':
    main()
