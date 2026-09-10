#!/usr/bin/env python3
"""Export public aggregates from CAP-SAT logs. Python stdlib only; no raw CNFs exported."""
import argparse
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 456, 789, 1024]
ARMS = ['capsat', 'default', 'polarity', 'walksat_equal']
ROOTS = {'frozen': 'e17_scale_ladder_v1', 'small': 'e19_indist_n350_v1', 'large': 'e19_indist_n350_large_v1'}
NAMES = {'frozen': 'E17 · Frozen', 'small': 'E19S · 500 train', 'large': 'E19L · 2,000 train'}

def read_run(root, seed):
    rows, errors, stamps = {}, [], []
    for path in sorted((root / 'solve_cadical').glob(f'n350_seed{seed}_cadical195_shard*.jsonl')):
        stamps.append(path.stat().st_mtime)
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('record is not an object')
                if row.get('record_type') != 'solve_row':
                    continue
                i = row['row_index']
                if type(i) is not int or not 0 <= i < 300:
                    raise ValueError('invalid row index')
                if row.get('arm_errors') or row.get('consistency_error'):
                    raise ValueError('solver error')
                if set(row['arms']) != set(ARMS):
                    raise ValueError('missing arm')
                for arm in row['arms'].values():
                    t = arm['full_pipeline_par2_s']
                    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
                        raise ValueError('invalid PAR2')
                    if arm.get('satisfiable') and arm.get('unsatisfiable'):
                        raise ValueError('contradictory outcome')
                if i in rows and rows[i] != row:
                    raise ValueError('conflicting duplicate')
                rows[i] = row
            except (ValueError, KeyError, TypeError):
                errors.append('invalid record')
    return rows, len(errors), max(stamps, default=None)

def interval(values):
    rng = random.Random(20260910)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [round(statistics.mean(values), 2), round(means[250], 2), round(means[9749], 2)]

def native_progress(repo):
    """Only public progress fields; omit process IDs, paths, commands and host data."""
    path = repo / 'experiments/native_baselines_20260910/SUPERVISION.json'
    try:
        raw = json.loads(path.read_text())
        progress = raw.get('progress', {})
        preparation = raw.get('preparation') or {}
        keys = ['completed_iterations','target_iterations','completed_epochs','target_epochs','label_records','stage_wallclock_cap_hours']
        return {'observed':raw.get('time_utc'), 'stage':raw.get('stage'),
                'progress':{k:progress[k] for k in keys if type(progress.get(k)) in (int,float) and math.isfinite(progress[k])},
                'controller_alive':raw.get('controller_alive') is True,
                'stage_alive':raw.get('stage_alive') is True,
                'alerts':[x for x in raw.get('alerts',[]) if isinstance(x,str) and all(c.isupper() or c.isdigit() or c=='_' for c in x)],
                'preparation':{k:preparation[k] for k in ['label_records','target_records'] if type(preparation.get(k)) is int},
                'scope':'缩减预算原生训练；正式测试对比尚未完成'}
    except (OSError,ValueError,TypeError,AttributeError):
        return None

def collect(repo):
    runs, truth, identities = [], {}, {}
    digest = hashlib.sha256()
    for variant, directory in ROOTS.items():
        root = repo / 'experiments/sprint_20260905' / directory
        for seed in SEEDS:
            rows, errors, stamp = read_run(root, seed)
            for i, row in rows.items():
                cnf = row['cnf_path']
                if i in identities and identities[i] != cnf:
                    raise ValueError('CNF identity mismatch')
                identities[i] = cnf
                digest.update(json.dumps(row, sort_keys=True).encode())
                for arm in row['arms'].values():
                    label = 'SAT' if arm.get('satisfiable') else 'UNSAT' if arm.get('unsatisfiable') else None
                    if label and i in truth and truth[i] != label:
                        raise ValueError('SAT/UNSAT contradiction')
                    if label:
                        truth[i] = label
            runs.append({'variant':variant, 'seed':seed, 'rows':rows, 'errors':errors, 'updated':stamp})
    results = []
    for variant in ROOTS:
        complete = [r for r in runs if r['variant'] == variant and len(r['rows']) == 300 and not r['errors']]
        for group in ['SAT', 'UNSAT', 'ALL']:
            indices = [i for i in range(300) if group == 'ALL' or truth.get(i) == group]
            for comparator in ARMS[1:]:
                if len(complete) < 2 or not indices:
                    continue
                values = [statistics.mean(r['rows'][i]['arms'][comparator]['full_pipeline_par2_s'] - r['rows'][i]['arms']['capsat']['full_pipeline_par2_s'] for i in indices) for r in complete]
                results.append({'variant':variant, 'group':group, 'comparator':comparator, 'n':len(indices), 'seeds':len(complete), 'gain':interval(values)})
    for r in runs:
        r['count'] = len(r.pop('rows'))
        r['complete'] = r['count'] == 300 and r['errors'] == 0
        r['updated'] = dt.datetime.fromtimestamp(r['updated'], dt.timezone.utc).isoformat() if r['updated'] else None
    return {'generated':dt.datetime.now(dt.timezone.utc).isoformat(), 'source_sha256':digest.hexdigest(), 'source':'E17/E19 n350 solve_row records', 'runs':runs, 'results':results, 'truth':{g:sum(truth.get(i,'UNKNOWN') == g for i in range(300)) for g in ['SAT','UNSAT','UNKNOWN']}, 'issues':json.loads((HERE / 'issues.json').read_text()), 'native':native_progress(repo)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=HERE.parent)
    parser.add_argument('--out', type=Path, default=HERE / 'site')
    args = parser.parse_args()
    data = collect(args.repo.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    template = (HERE / 'template.html').read_text()
    page = template.replace('{{DATA}}', payload.replace('<', '\\u003c')).replace('{{SHA}}', data['source_sha256']).replace('{{TIME}}', html.escape(data['generated']))
    for name, content in [('status.json',payload), ('index.html',page)]:
        temp = args.out / (name + '.tmp')
        temp.write_text(content, encoding='utf-8')
        temp.replace(args.out / name)
    print(f"Exported {sum(r['count'] for r in data['runs'])}/4500 rows; {sum(r['complete'] for r in data['runs'])}/15 complete seeds → {args.out}")

if __name__ == '__main__':
    main()
