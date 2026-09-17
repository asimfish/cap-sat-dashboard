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
    p.add_argument('--stage-only', action='store_true', help='Stage the allowlist for review without committing or pushing')
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
        unchanged = old['source_sha256'] == current['source_sha256'] and old['issues'] == current['issues'] and old.get('native') == current.get('native') and old.get('native_first') == current.get('native_first') and old.get('comparison') == current.get('comparison') and old.get('performance_plan') == current.get('performance_plan') and old.get('pilot') == current.get('pilot') and old.get('hybrid') == current.get('hybrid') and old.get('utility') == current.get('utility') and old.get('conservative') == current.get('conservative')
        same_code = all((dest/n).exists() and (dest/n).read_bytes() == (source/n).read_bytes() for n in ['build.py','template.html','issues.json','performance_plan.json','README.md','publish.py','watch.py','test_build.py','test_paired_launch.py','EVIDENCE.md','site/evidence.html','LICENSE','.github/workflows/pages.yml'])
        unchanged = unchanged and old.get('cost_aware') == current.get('cost_aware')
        unchanged = unchanged and old.get('shared_gpu') == current.get('shared_gpu')
        unchanged = unchanged and old.get('structured_decoder') == current.get('structured_decoder') and old.get('structured_policy') == current.get('structured_policy')
        unchanged = unchanged and old.get('resident_policy') == current.get('resident_policy')
        unchanged = unchanged and old.get('decoder_utility') == current.get('decoder_utility')
        unchanged = unchanged and old.get('stability') == current.get('stability')
        unchanged = unchanged and old.get('overnight') == current.get('overnight')
        unchanged = unchanged and old.get('prefix_campaign') == current.get('prefix_campaign')
        unchanged = unchanged and old.get('joint_feedback') == current.get('joint_feedback')
        unchanged = unchanged and old.get('local_feedback') == current.get('local_feedback')
        unchanged = unchanged and old.get('compact_distill') == current.get('compact_distill')
        unchanged = unchanged and old.get('boundary_screen') == current.get('boundary_screen')
        unchanged = unchanged and old.get('coverage_screen') == current.get('coverage_screen')
        unchanged = unchanged and old.get('night_recovery') == current.get('night_recovery')
        unchanged = unchanged and old.get('depth_screen') == current.get('depth_screen')
        unchanged = unchanged and old.get('literal_fullcost') == current.get('literal_fullcost')
        unchanged = unchanged and old.get('weight_precision') == current.get('weight_precision')
        unchanged = unchanged and old.get('head_refit') == current.get('head_refit')
        unchanged = unchanged and old.get('strategy_headroom') == current.get('strategy_headroom')
        unchanged = unchanged and old.get('targeted') == current.get('targeted')
        unchanged = unchanged and old.get('exact_search') == current.get('exact_search')
        unchanged = unchanged and old.get('confirmation') == current.get('confirmation')
        unchanged = unchanged and old.get('recognition_repair') == current.get('recognition_repair')
        if unchanged and same_code and age < 3600:
            print('No change; next freshness export within one hour')
            return
    names = ['build.py','template.html','issues.json','performance_plan.json','README.md','publish.py','watch.py','test_build.py','test_paired_launch.py','LICENSE','site/index.html','site/status.json','site/evidence.html','EVIDENCE.md','.github/workflows/pages.yml']
    for name in names:
        src = source/name
        if src.exists():
            (dest/name).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest/name)
    run(['git','add','--']+[n for n in names if (dest/n).exists()],dest)
    if a.stage_only:
        print('Dashboard allowlist staged; commit and push withheld for review')
        return
    if run(['git','diff','--cached','--name-only'],dest):
        run(['git','diff','--cached','--check'],dest)
        run(['git','commit','-m','[dashboard/chore]: refresh audited CAP-SAT progress'],dest)
        run(['git','push','origin','main'],dest)
    print('Dashboard published')

if __name__ == '__main__':
    main()
