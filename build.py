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
import re
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

def verify_terminal(root, raw):
    """Recheck accepted immutable artifacts; a stopped monitor is not a stale run."""
    if raw.get('terminal') is not True:
        return 'not_terminal', {}
    try:
        for name in ['rlaf_verified_v2','neuroback_verified_v2']:
            accepted = raw['acceptance'][name]
            receipt = json.loads((root/(name+'_RECEIPT.json')).read_text())
            if accepted['status'] != 'training_artifacts_verified' or receipt['returncode'] != 0 or accepted.get('errors'):
                return 'not_verified', {}
            hashes = accepted['hashes']
            required = {'rlaf_verified_v2.log','rlaf_verified_v2/best.pt'} if name.startswith('rlaf') else {'neuroback_small/training.jsonl','neuroback_small/COMPLETE.json','neuroback_small/best.ptg'}
            if not required.issubset(hashes):
                return 'not_verified', {}
            for relative, expected in hashes.items():
                path = (root/relative).resolve()
                if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    return 'not_verified', {}
        iterations = len(re.findall(r'Optimized model for 50 steps', (root/'rlaf_verified_v2.log').read_text()))
        epochs = [json.loads(line)['epoch'] for line in (root/'neuroback_small/training.jsonl').read_text().splitlines()]
        if iterations != 100 or epochs != list(range(40)):
            return 'not_verified', {}
        return 'verified', {'completed_iterations':iterations,'target_iterations':100,'completed_epochs':len(epochs),'target_epochs':40}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return 'not_verified', {}

def native_progress(repo):
    """Only public progress fields; omit process IDs, paths, commands and host data."""
    path = repo / 'experiments/native_baselines_20260910/SUPERVISION.json'
    try:
        raw = json.loads(path.read_text())
        progress = raw.get('progress', {})
        terminal_integrity, completed = verify_terminal(path.parent, raw)
        progress = dict(progress, **completed)
        preparation = raw.get('preparation') or {}
        keys = ['completed_iterations','target_iterations','completed_epochs','target_epochs','label_records','stage_wallclock_cap_hours']
        return {'observed':raw.get('time_utc'), 'stage':raw.get('stage'),
                'terminal_integrity':terminal_integrity,
                'phase':raw.get('phase') if raw.get('phase') in ['training','waiting_for_gpu','label_preparing','terminal','queued'] else None,
                'progress':{k:progress[k] for k in keys if type(progress.get(k)) in (int,float) and math.isfinite(progress[k])},
                'controller_alive':raw.get('controller_alive') is True,
                'stage_alive':raw.get('stage_alive') is True,
                'alerts':[x for x in raw.get('alerts',[]) if isinstance(x,str) and all(c.isupper() or c.isdigit() or c=='_' for c in x)],
                'preparation':{k:preparation[k] for k in ['label_records','target_records'] if type(preparation.get(k)) is int},
                'acceptance':{k:v.get('status') for k,v in (raw.get('acceptance') or {}).items() if k in ['rlaf_verified_v2','neuroback_verified_v2'] and isinstance(v,dict) and v.get('status') in ['pending','not_accepted_nonzero_exit','failed_acceptance','training_artifacts_verified']},
                'scope':'缩减预算原生训练；正式测试与验收状态见原生比较面板'}
    except (OSError,ValueError,TypeError,AttributeError):
        return None

def comparison_progress(repo):
    root=repo/'experiments/native_comparison_20260911_v1'
    try:
        raw=json.loads((root/'SUPERVISION.json').read_text())
        keys=['completed_paired_rows','target_paired_rows','completed_arm_cells','target_arm_cells']
        result={k:raw[k] for k in keys if type(raw.get(k)) is int and raw[k]>=0}
        result['observed']=dt.datetime.fromtimestamp(raw['time_unix'],dt.timezone.utc).isoformat()
        phase=raw.get('state',{}).get('phase')
        result['phase']=phase if phase in ['solving','aggregate','terminal','predict_capsat','predict_rlaf','predict_neuroback','waiting_for_gpu_resources'] else 'unknown'
        result['alerts']=[x for x in raw.get('alerts',[]) if isinstance(x,str) and all(c.isupper() or c.isdigit() or c=='_' for c in x)]
        result['summary_status']=None
        if (root/'SUMMARY.json').exists():
            summary=json.loads((root/'SUMMARY.json').read_text())
            if summary.get('status') in ['complete_pending_owner_review','all_cells_recorded_with_evidence_gaps']:
                result['summary_status']=summary['status']
                result['failed_cells']=len(summary['failures'])
                result['unverified_unsat_cells']=summary['unverified_unsat_cells']
        result['audit']=comparison_audit(repo)
        return result
    except (OSError,ValueError,KeyError,TypeError):
        return None

def comparison_audit(repo):
    """Publish allowlisted aggregates only; keep the original verification policy."""
    path=repo/'iclr_2027/audit/native_comparison_20260911_v1/TERMINAL_AUDIT.json'
    def number(value):
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('invalid audit number')
        return value
    try:
        payload=path.read_bytes()
        raw=json.loads(payload)
        if raw['complete_denominator'] is not True or raw['completed_rows']!=619 or raw['recorded_cells']!=7428:
            return None
        rows=[]
        for cohort in ['fresh_n200','existing_n350','existing_industrial19']:
            for stratum,group in raw['groups'][cohort].items():
                if stratum not in ['ALL','SAT','UNSAT','UNKNOWN']:
                    continue
                if group['n']==0:
                    continue
                for backend in ['glucose','kissat']:
                    for arm in ['stock','native','capsat_standing_adapter','polarity_initial','polarity_standing','walksat_standing']:
                        source=group['backends'][backend][arm]
                        row=dict(cohort=cohort,stratum=stratum,backend=backend,arm=arm)
                        for key in ['n','verified_solved','unverified_unsat','reported_par2_s','verified_par2_s']:
                            row[key]=number(source[key])
                        for policy in ['reported','verified']:
                            contrast=source.get(policy+'_contrast')
                            if contrast:
                                ci=contrast['ci95']
                                if len(ci)!=2: raise ValueError('invalid interval')
                                row[policy+'_contrast']={ 'mean':number(contrast['mean']), 'ci95':[number(x) for x in ci] }
                        rows.append(row)
        return {'sha256':hashlib.sha256(payload).hexdigest(),'observed':dt.datetime.fromtimestamp(raw['observed_unix'],dt.timezone.utc).isoformat(),
                'issues':len(raw['issues']),'failures':len(raw['failures']),
                'unverified_unsat_cells':len(raw['unverified_unsat_cells']),'rows':rows}
    except (OSError,ValueError,KeyError,TypeError):
        return None

PILOT_ARMS=['stock','segmented','cap_continuous','cap_pulse','pol_pulse','ws_pulse']

def pilot_progress(repo):
    """Public aggregate only: never export paths, phases, host identity or logs."""
    root=repo/'experiments/performance_pilot_20260912_e26'
    try:
        raw=(root/'STATUS.json').read_bytes();status=json.loads(raw)
        frozen=(root/'FROZEN.json').read_bytes();cfg=json.loads(frozen)
        if status['phase'] not in ['running','terminal','failed']:
            raise ValueError('unknown pilot phase')
        def count(value,maximum):
            if type(value) is not int or not 0<=value<=maximum:
                raise ValueError('invalid pilot count')
            return value
        def number(value):
            if type(value) not in [int,float] or not math.isfinite(value) or value<0:
                raise ValueError('invalid nonnegative pilot metric')
            return value
        if status['target_rows']!=24 or status['target_cells']!=144 or cfg['arms']!=PILOT_ARMS or cfg['cutoff_wall_s']!=10:
            raise ValueError('unexpected pilot design')
        result={'id':'E26','phase':status['phase'],'observed':status.get('observed',status['started']),
                'completed_rows':count(status['completed_rows'],24),'target_rows':24,
                'completed_cells':count(status['completed_cells'],144),'target_cells':144,
                'cutoff_wall_s':number(cfg['cutoff_wall_s']),
                'frozen_sha256':hashlib.sha256(frozen).hexdigest(),
                'status_sha256':hashlib.sha256(raw).hexdigest(),'readout':None}
        dt.datetime.fromisoformat(result['observed'])
        if status['phase']=='terminal':
            if result['completed_rows']!=24 or result['completed_cells']!=144:
                raise ValueError('incomplete terminal pilot')
            if not (root/'RESULTS.json').exists():
                result['phase']='awaiting_readout';return result
            payload=(root/'RESULTS.json').read_bytes();report=json.loads(payload)
            if report['frozen_sha256']!=result['frozen_sha256'] or report['rows']!=24 or report['cells']!=144:
                raise ValueError('pilot readout identity mismatch')
            if report['status'] not in ['promising_requires_independent_confirmation','no_confirmed_positive_effect']:
                raise ValueError('unknown pilot verdict')
            summaries={}
            for arm in PILOT_ARMS:
                s=report['summaries'][arm]
                summaries[arm]={k:number(s[k]) for k in ['verified_par2_s','reported_par2_s']}
                if any(v>20 for v in summaries[arm].values()):raise ValueError('pilot metric beyond PAR2 bound')
                summaries[arm].update({k:count(s[k],24) for k in ['verified_solved','entered','released']})
            contrasts={}
            for arm in ['stock','segmented','pol_pulse','ws_pulse']:
                c=report['contrasts'][arm];band=c['simultaneous95_band']
                if len(band)!=2 or type(c['gate']) is not bool:raise ValueError('invalid pilot contrast')
                # Gains/intervals are signed; number() intentionally rejects negatives.
                signed=[c['mean_gain_s'],*band]
                if any(type(v) not in [int,float] or not math.isfinite(v) for v in signed):raise ValueError('invalid signed metric')
                if band[0]>band[1]:raise ValueError('reversed interval')
                contrasts[arm]={'mean_gain_s':signed[0],'simultaneous95_band':band,'gate':c['gate']}
            if report['status']=='promising_requires_independent_confirmation' and (
                    report['errors'] or report['walksat_budget_failures'] or not all(c['gate'] for c in contrasts.values())):
                raise ValueError('positive verdict contradicts gates')
            result['readout']={'status':report['status'],'summaries':summaries,'contrasts':contrasts,
                'errors':len(report['errors']),'walksat_budget_failures':len(report['walksat_budget_failures']),
                'unverified_unsat_cells':count(report['unverified_unsat_cells'],144),
                'sha256':hashlib.sha256(payload).hexdigest()}
        return result
    except (OSError,ValueError,KeyError,TypeError):
        return None

HYBRID_ARMS=['stock','random_repair','polarity_repair','cap_warm_repair','cap_soft_repair']

def hybrid_progress(repo):
    root=repo/'experiments/performance_hybrid_20260912_e27'
    def num(v,lo=0,hi=20):
        if type(v) not in [int,float] or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('invalid hybrid metric')
        return v
    def count(v,maximum):
        if type(v) is not int:raise ValueError('invalid count type')
        return num(v,0,maximum)
    def cells(raw,arms,n):
        out={}
        for a in arms:
            s=raw[a];out[a]={k:num(s[k]) for k in ['verified_par2_s','reported_par2_s']}
            out[a].update({k:count(s[k],n) for k in ['verified_solved','sls_solved','fallback_solved']})
            if out[a]['sls_solved']+out[a]['fallback_solved']!=out[a]['verified_solved']:raise ValueError('solve-source mismatch')
        return out
    try:
        result={'id':'E27','stages':{},'selection':None}
        for stage,n in [('dev',24),('test',48)]:
            folder=root/stage
            if not (folder/'STATUS.json').exists():continue
            frozen=(folder/'FROZEN.json').read_bytes();cfg=json.loads(frozen)
            status=json.loads((folder/'STATUS.json').read_text());arms=cfg['arms']
            if stage=='test' and cfg['selection_sha256']!=hashlib.sha256((root/'SELECTION.json').read_bytes()).hexdigest():raise ValueError('selection identity drift')
            if not (arms==HYBRID_ARMS if stage=='dev' else arms[:3]==HYBRID_ARMS[:3] and len(arms) in [3,4] and set(arms)<=set(HYBRID_ARMS)):
                raise ValueError('unexpected hybrid arms')
            if status['phase'] not in ['running','failed','terminal'] or status['target_rows']!=n or status['target_cells']!=n*len(arms):raise ValueError('unexpected hybrid stage')
            h=hashlib.sha256(frozen).hexdigest()
            if status['frozen_sha256']!=h:raise ValueError('frozen identity drift')
            out={'phase':status['phase'],'observed':status['observed'],'completed_rows':count(status['completed_rows'],n),
                 'completed_cells':count(status['completed_cells'],n*len(arms)),'target_rows':n,'target_cells':n*len(arms),'frozen_sha256':h,'readout':None}
            dt.datetime.fromisoformat(out['observed'])
            if out['phase']=='terminal':
                if out['completed_rows']!=n or out['completed_cells']!=n*len(arms):raise ValueError('incomplete terminal stage')
                if not (folder/'RESULTS.json').exists():out['phase']='awaiting_readout'
                else:
                    payload=(folder/'RESULTS.json').read_bytes();r=json.loads(payload)
                    if r['frozen_sha256']!=h or r['prepared_sha256']!=status['prepared_sha256'] or r['rows']!=n or r['cells']!=n*len(arms):raise ValueError('hybrid readout identity mismatch')
                    if r['status'] not in ['development_complete','no_confirmed_improvement','local_engineering_improvement_only','local_learning_improvement']:raise ValueError('unknown hybrid verdict')
                    contrasts={}
                    for key,c in r['contrasts'].items():
                        if key not in ['engineering_vs_stock','cap_vs_stock','cap_vs_random_repair','cap_vs_polarity_repair']:raise ValueError('unknown contrast')
                        if c['arm'] not in arms or c['control'] not in arms or type(c['gate']) is not bool:raise ValueError('bad contrast arms')
                        band=c['simultaneous95_band']
                        if len(band)!=2 or band[0]>band[1]:raise ValueError('bad interval')
                        contrasts[key]={'arm':c['arm'],'control':c['control'],'mean_gain_s':num(c['mean_gain_s'],-20,20),'simultaneous95_band':[num(v,-20,20) for v in band],'gate':c['gate']}
                    if type(r['engineering_pass']) is not bool or type(r['learning_pass']) is not bool:raise ValueError('bad gate type')
                    if r['engineering_pass'] and (r['errors'] or not contrasts.get('engineering_vs_stock',{}).get('gate')):raise ValueError('unjustified engineering pass')
                    if r['learning_pass'] and (r['errors'] or not all(contrasts.get('cap_vs_'+a,{}).get('gate') for a in HYBRID_ARMS[:3])):raise ValueError('unjustified learning pass')
                    expected='development_complete' if stage=='dev' else 'local_learning_improvement' if r['learning_pass'] else 'local_engineering_improvement_only' if r['engineering_pass'] else 'no_confirmed_improvement'
                    if r['status']!=expected or stage=='dev' and (r['learning_pass'] or r['engineering_pass']):raise ValueError('verdict/stage mismatch')
                    out['readout']={'status':r['status'],'engineering_pass':r['engineering_pass'],'learning_pass':r['learning_pass'],
                        'summaries':cells(r['summaries'],arms,n),'by_scale':{s:cells(r['by_scale'][s],arms,n//2) for s in ['325','500']},
                        'contrasts':contrasts,'errors':len(r['errors']),'unverified_unsat_cells':count(r['unverified_unsat_cells'],n*len(arms)),
                        'sha256':hashlib.sha256(payload).hexdigest()}
                    if stage=='test':
                        public=out['readout'];public['qualification']='awaiting_matched_guard'
                        public['original_engineering_pass']=public['engineering_pass'];public['original_learning_pass']=public['learning_pass']
                        public['engineering_pass']=False;public['learning_pass']=False
                        if (root/'JOINT_RESULTS.json').exists():
                            joint_payload=(root/'JOINT_RESULTS.json').read_bytes();j=json.loads(joint_payload)
                            if j['main_results_sha256']!=public['sha256'] or j['matched_frozen_sha256']!=hashlib.sha256((root/'matched/FROZEN.json').read_bytes()).hexdigest():raise ValueError('joint identity mismatch')
                            if j['matched_executed'] and j['matched_results_sha256']!=hashlib.sha256((root/'matched/RESULTS.json').read_bytes()).hexdigest():raise ValueError('matched readout drift')
                            matches=['polarity_warm_match','random_warm_match'];controls=HYBRID_ARMS[:3]+matches
                            jc={}
                            for key,c in j['contrasts'].items():
                                if key not in ['engineering_vs_stock',*['cap_vs_'+a for a in controls]] or c['arm'] not in arms or c['control'] not in [*arms,*matches] or type(c['gate']) is not bool:raise ValueError('invalid joint contrast')
                                band=c['simultaneous95_band']
                                if len(band)!=2 or band[0]>band[1]:raise ValueError('invalid joint interval')
                                jc[key]={'arm':c['arm'],'control':c['control'],'mean_gain_s':num(c['mean_gain_s'],-20,20),'simultaneous95_band':[num(v,-20,20) for v in band],'gate':c['gate']}
                            if type(j['learning_pass']) is not bool or type(j['engineering_pass']) is not bool:raise ValueError('invalid joint gate')
                            if j['engineering_pass'] and not (public['original_engineering_pass'] and jc.get('engineering_vs_stock',{}).get('gate')):raise ValueError('invalid engineering promotion')
                            if j['learning_pass'] and not (public['original_learning_pass'] and j['matched_executed'] and not j['errors'] and all(jc.get('cap_vs_'+a,{}).get('gate') for a in controls)):raise ValueError('missing exact matched controls')
                            public.update(qualification='joint_guard_complete',engineering_pass=j['engineering_pass'],learning_pass=j['learning_pass'],contrasts=jc,joint_sha256=hashlib.sha256(joint_payload).hexdigest(),matched_executed=bool(j['matched_executed']))
                            public['status']='local_learning_improvement' if j['learning_pass'] else 'local_engineering_improvement_only' if j['engineering_pass'] else 'no_confirmed_improvement'
                            if j['matched_executed']:
                                public['summaries'].update(cells(j['extra_summaries'],matches,n))
                                for scale in ['325','500']:public['by_scale'][scale].update(cells(j['extra_by_scale'][scale],matches,n//2))
            result['stages'][stage]=out
        if (root/'SELECTION.json').exists():
            raw=(root/'SELECTION.json').read_bytes();s=json.loads(raw)
            if s['engineering'] not in [None,*HYBRID_ARMS[1:3]] or s['cap'] not in [None,*HYBRID_ARMS[3:]]:raise ValueError('unknown selected arm')
            result['selection']={'engineering':s['engineering'],'cap':s['cap'],'sha256':hashlib.sha256(raw).hexdigest()}
        return result if result['stages'] else None
    except (OSError,ValueError,TypeError,KeyError):return None

def utility_progress(repo):
    """E28 allowlist: no raw trajectories, weights, probabilities or host metadata."""
    root=repo/'experiments/utility_ranker_20260913_e28'
    def metric(value, low=0, high=10):
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('invalid utility metric')
        return value
    try:
        if not (root/'DATA_FROZEN.json').exists():return None
        result={'data_frozen_sha256':hashlib.sha256((root/'DATA_FROZEN.json').read_bytes()).hexdigest(),'training':None,'stages':{},'selection':None}
        if (root/'TRAINED.json').exists():
            raw=(root/'TRAINED.json').read_bytes();trained=json.loads(raw)
            if trained['data_frozen_sha256']!=result['data_frozen_sha256']:raise ValueError('training identity')
            result['training']={'train_groups':metric(trained['train_groups'],0,4608),'validation_groups':metric(trained['validation_groups'],0,1536),
                'models':{name:{'proxy_gain':metric(trained['models'][name]['validation_gain_vs_minbreak'],-1000,1000),
                          'parameters':metric(trained['models'][name]['parameters'],723,723)} for name in ['utility_cheap','utility_cap']},
                'sha256':hashlib.sha256(raw).hexdigest()}
        for stage,n in [('dev',24),('test',48)]:
            folder=root/stage
            if not (folder/'STATUS.json').exists():continue
            frozen=(folder/'FROZEN.json').read_bytes();cfg=json.loads(frozen);h=hashlib.sha256(frozen).hexdigest()
            state=json.loads((folder/'STATUS.json').read_text());arms=cfg['arms']
            if len(set(arms))!=len(arms) or not set(arms)<=set(['stock','random_repair','utility_cheap','utility_cap']):raise ValueError('utility arms')
            if state['frozen_sha256']!=h or state['target_rows']!=n or state['target_cells']!=n*len(arms):raise ValueError('utility identity')
            if state['phase'] not in ['running','failed','terminal']:raise ValueError('utility state')
            x={'phase':state['phase'],'observed':state['observed'],'target_rows':n,'target_cells':n*len(arms),
               'completed_rows':metric(state['completed_rows'],0,n),'completed_cells':metric(state['completed_cells'],0,n*len(arms)),
               'frozen_sha256':h,'readout':None}
            if state['phase']=='terminal':
                if x['completed_rows']!=n or x['completed_cells']!=x['target_cells']:raise ValueError('partial terminal')
                x['phase']='awaiting_readout'
                if (folder/'RESULTS.json').exists():
                    raw=(folder/'RESULTS.json').read_bytes();r=json.loads(raw)
                    if r['frozen_sha256']!=h or r['prepared_sha256']!=state['prepared_sha256'] or r['rows']!=n or r['cells']!=x['target_cells']:raise ValueError('utility readout identity')
                    if type(r['confirmed']) is not bool or (stage=='dev' and r['confirmed']):raise ValueError('development is not confirmation')
                    if set(r['summaries'])!=set(arms):raise ValueError('utility summaries')
                    summaries={a:{'solved':metric(r['summaries'][a]['solved'],0,n),'par2_s':metric(r['summaries'][a]['par2_s']),
                                  'sls_solved':metric(r['summaries'][a]['sls_solved'],0,r['summaries'][a]['solved'])} for a in arms}
                    if stage=='test':
                        if cfg['selection_sha256']!=hashlib.sha256((root/'SELECTION.json').read_bytes()).hexdigest():raise ValueError('selection drift')
                    if r['confirmed'] and (r['errors'] or not r['contrasts'] or not all(c['gate'] for c in r['contrasts'].values())):raise ValueError('unsubstantiated confirmation')
                    x.update(phase='terminal',readout={'summaries':summaries,'confirmed':r['confirmed'],'errors':len(r['errors']),
                        'sha256':hashlib.sha256(raw).hexdigest()})
            result['stages'][stage]=x
        if (root/'SELECTION.json').exists():
            selection=json.loads((root/'SELECTION.json').read_text());name=selection['selected']
            if name not in [None,'utility_cheap','utility_cap']:raise ValueError('unknown utility selection')
            if selection['development_sha256']!=hashlib.sha256((root/'dev/RESULTS.json').read_bytes()).hexdigest():raise ValueError('selection source')
            result['selection']={'selected':name,'stopped':name is None}
        return result
    except (OSError,ValueError,TypeError,KeyError):return None

def conservative_progress(repo, *, cost_aware=False):
    """E29/E30 allowlisted aggregates; search trials are not independent instances."""
    root=repo/('experiments/cost_aware_20260913_e30' if cost_aware else 'experiments/conservative_search_20260913_e29')
    cells,workers,trials=(9216,16,128) if cost_aware else (18432,32,192)
    score,score_limit=('fitness_s',.5) if cost_aware else ('fitness',400000)
    learned=['cost_cheap','cost_cap'] if cost_aware else ['learned_cheap','learned_cap']
    allowed=['stock','random','hand']+(['e29_cheap','e29_cap'] if cost_aware else [])+learned
    test_arms=[allowed[:5]+['cost_cheap'],allowed[:5]+['cost_cap','cost_cheap']] if cost_aware else [allowed,allowed[:-1]]
    def num(x,lo=0,hi=10,integer=False):
        if type(x) not in (int,float) or not math.isfinite(x) or not lo<=x<=hi or (integer and type(x) is not int):raise ValueError('E29 metric')
        return x
    def sm(value,arms,n):
        if set(value)!=set(arms):raise ValueError('E29 arms')
        return {a:{'solved':num(value[a]['solved'],0,n,True),'par2_s':num(value[a]['par2_s']),
            'sls_solved':num(value[a]['sls_solved'],0,value[a]['solved'],True)} for a in arms}
    try:
        if not (root/'TRAIN_STATUS.json').exists():return None
        raw=(root/'train/FROZEN.json').read_bytes();h=hashlib.sha256(raw).hexdigest();state=json.loads((root/'TRAIN_STATUS.json').read_text())
        if state['frozen_sha256']!=h or state['phase'] not in ['running','failed','terminal'] or state['target_cells']!=cells:raise ValueError('training identity')
        train={'phase':state['phase'],'observed':state['observed'],'completed_cells':num(state['completed_cells'],0,cells,True),'target_cells':cells,
            'workers':num(state['workers'],1,workers,True),'frozen_sha256':h,'readout':None}
        if state['phase']=='terminal':
            if state['completed_cells']!=cells:raise ValueError('partial training terminal')
            raw=(root/'TRAINED.json').read_bytes();trained=json.loads(raw)
            if trained['train_frozen_sha256']!=h or trained['training_cells']!=cells:raise ValueError('trained identity')
            train['readout']={'sha256':hashlib.sha256(raw).hexdigest(),'summaries':{a:{('fitness_s' if cost_aware else 'fitness_flips'):num(trained['selected'][a][score],0,score_limit),
                'solved':num(trained['selected'][a]['solved'],0,trials,True),'trials':num(trained['selected'][a]['trials'],trials,trials,True)} for a in ['cheap','cap']}}
        result={'training':train,'stages':{},'selection':None}
        for stage,instances in [('dev',96),('test',192)]:
            folder=root/stage
            if not (folder/'STATUS.json').exists():continue
            raw=(folder/'FROZEN.json').read_bytes();cfg=json.loads(raw);h=hashlib.sha256(raw).hexdigest();s=json.loads((folder/'STATUS.json').read_text());arms=cfg['arms'];n=instances*3
            if (stage=='dev' and arms!=allowed) or (stage=='test' and arms not in test_arms):raise ValueError('stage arms')
            if s['frozen_sha256']!=h or s['phase'] not in ['running','failed','terminal'] or s['instances']!=instances or s['target_rows']!=n or s['target_cells']!=n*len(arms):raise ValueError('E29 stage identity')
            x={'phase':s['phase'],'observed':s['observed'],'instances':instances,'target_rows':n,'target_cells':n*len(arms),
                'completed_rows':num(s['completed_rows'],0,n,True),'completed_cells':num(s['completed_cells'],0,n*len(arms),True),'frozen_sha256':h,'readout':None}
            if s['phase']=='terminal':
                if s['completed_rows']!=n or s['completed_cells']!=n*len(arms):raise ValueError('partial E29 terminal')
                x['phase']='awaiting_readout'
                if (folder/'RESULTS.json').exists():
                    raw=(folder/'RESULTS.json').read_bytes();r=json.loads(raw)
                    if r['frozen_sha256']!=h or r['prepared_sha256']!=s['prepared_sha256'] or r['instances']!=instances or r['trials']!=n or r['cells']!=n*len(arms):raise ValueError('E29 readout identity')
                    if type(r['confirmed']) is not bool or (stage=='dev' and r['confirmed']):raise ValueError('E29 false confirmation')
                    if stage=='test' and cfg['selection_sha256']!=hashlib.sha256((root/'SELECTION.json').read_bytes()).hexdigest():raise ValueError('E29 selection drift')
                    if r['confirmed'] and (r['errors'] or not r['contrasts'] or not all(c['gate'] for c in r['contrasts'].values())):raise ValueError('unsupported E29 gate')
                    x.update(phase='terminal',readout={'sha256':hashlib.sha256(raw).hexdigest(),'errors':len(r['errors']),'confirmed':r['confirmed'],
                        'summaries':sm(r['summaries'],arms,n),'by_scale':{z:sm(r['by_scale'][z],arms,n//2) for z in ['325','500']}})
                    if cost_aware:
                        if type(r['cap_active']) is not bool:raise ValueError('CAP gate type')
                        x['readout']['cap_active']=r['cap_active']
            result['stages'][stage]=x
        if (root/'SELECTION.json').exists():
            s=json.loads((root/'SELECTION.json').read_text())
            if s['selected'] not in [None]+learned or s['development_sha256']!=hashlib.sha256((root/'dev/RESULTS.json').read_bytes()).hexdigest():raise ValueError('selection')
            result['selection']={'selected':s['selected'],'stopped':s['selected'] is None}
        if cost_aware and (root/'PREFLIGHT.json').exists():
            pre=json.loads((root/'PREFLIGHT.json').read_text())
            if (pre['reason']!='both_candidates_identical_to_hand' or pre['decision']!='stop_before_development_no_distinct_candidate'
                or pre['dev_generated'] is not False or pre['test_generated'] is not False or pre['protocol_deviation'] is not True
                or pre['cap_active'] is not False or train['phase']!='terminal'
                or pre['trained_sha256']!=train['readout']['sha256'] or result['stages'] or result['selection']
                or (root/'dev').exists() or (root/'test').exists()
                or pre['audit_sha256']!=hashlib.sha256((root/'AUDIT.json').read_bytes()).hexdigest()):raise ValueError('cost identity stop')
            audit=json.loads((root/'AUDIT.json').read_text())
            if audit['verdict']!='integrity_pass_candidates_identical_to_hand' or audit['trained_sha256']!=pre['trained_sha256'] or audit['training_cells']!=9216:raise ValueError('cost audit')
            result['preflight']={'reason':'both_candidates_identical_to_hand','protocol_deviation':True,'dev_generated':False,'test_generated':False,
                                 'audit_sha256':pre['audit_sha256'],'checked_training_sat':num(audit['sat_witnesses_rechecked'],0,9216,True)}
        return result
    except (OSError,ValueError,KeyError,TypeError):return None

def shared_gpu_progress(repo):
    """E32 own-work aggregates, never publish other users' process inventory."""
    root=repo/'experiments/shared_gpu_20260913_e32'
    def number(x,lo=0,hi=100000):
        if type(x) not in (int,float) or not math.isfinite(x) or not lo<=x<=hi:raise ValueError('GPU metric')
        return x
    try:
        if not (root/'FROZEN.json').exists():return None
        raw=(root/'FROZEN.json').read_bytes();cfg=json.loads(raw);h=hashlib.sha256(raw).hexdigest()
        if cfg['seeds']!=[42,43,44] or cfg['epochs']!=120 or cfg['batch_size']!=16:raise ValueError('GPU contract')
        result={'frozen_sha256':h,'workers':{},'benchmark':None,'telemetry':{},'terminal':False}
        if (root/'BENCHMARK.json').exists():
            raw=(root/'BENCHMARK.json').read_bytes();b=json.loads(raw)
            if b['graphs']!=8 or b['warmups']!=3 or b['gradients_match'] is not True or b['batching_sha256']!=cfg['hashes']['experiments/shared_gpu_20260913_e32/batching.py']:raise ValueError('GPU benchmark identity')
            result['benchmark']={'sha256':hashlib.sha256(raw).hexdigest(),'summaries':{a:{k:number(b['summaries'][a][k],0,10000) for k in ['median_s','stdev_s','graphs_per_s']} for a in ['serial','batched']},'ratio':number(b['ratio'],0,100)}
        for seed in cfg['seeds']:
            folder=root/f'seed-{seed}'
            if not (folder/'STATUS.json').exists():continue
            s=json.loads((folder/'STATUS.json').read_text())
            if s['frozen_sha256']!=h or s['seed']!=seed or s['phase'] not in ['starting','running','failed','terminal'] or type(s['epoch']) is not int:raise ValueError('GPU worker identity')
            x={'phase':s['phase'],'observed':s['observed'],'epoch':number(s['epoch'],0,120),'target_epochs':120,'gpu_index':number(s['gpu_index'],0,7)}
            for k in ['train_loss','val_loss','best_val_loss','initial_validation_loss','peak_allocated_mib','reserved_mib','elapsed_s']:
                if k in s:x[k]=number(s[k])
            if s['phase']=='terminal':
                if s['epoch']!=120 or any(s[f'{n}_sha256']!=hashlib.sha256((folder/f'{n}.pt').read_bytes()).hexdigest() for n in ['best','last']):raise ValueError('GPU terminal integrity')
            result['workers'][str(seed)]=x
        if (root/'GPU_SAMPLES.json').exists():
            samples=json.loads((root/'GPU_SAMPLES.json').read_text())
            for seed,x in result['workers'].items():
                values=[number(d['utilization'],0,100) for sample in samples if sample.get('worker_alive',{}).get(seed) is True for d in sample['devices'] if d['index']==x['gpu_index']]
                if values:result['telemetry'][seed]={'samples':len(values),'card_utilization_mean':sum(values)/len(values),'card_utilization_max':max(values)}
        if (root/'TERMINAL.json').exists():
            t=json.loads((root/'TERMINAL.json').read_text())
            if t['frozen_sha256']!=h or type(t['completed']) is not bool or set(t['exit_codes'])!={'42','43','44'}:raise ValueError('GPU supervisor identity')
            if t['completed'] and any(type(v) is not int or v!=0 for v in t['exit_codes'].values()):raise ValueError('GPU supervisor exit')
            result['terminal']=t['completed'] and set(result['workers'])=={'42','43','44'} and all(x['phase']=='terminal' for x in result['workers'].values())
        replay=root/'resource-replay'
        if (replay/'RESULT.json').exists():
            r=json.loads((replay/'RESULT.json').read_text());rf=(replay/'REPLAY_FROZEN.json').read_bytes();rs=(replay/'PROCESS_SAMPLES.json').read_bytes()
            if r['completed'] is not True or r['original_process_memory_still_unmeasured'] is not True or r['scope']!='separate_identical_workload_resource_replay':raise ValueError('GPU replay scope')
            if r['exit_codes']!={'42':0,'43':0,'44':0} or json.loads(rf)['original_frozen_sha256']!=h or hashlib.sha256(rf).hexdigest()!=r['replay_frozen_sha256'] or hashlib.sha256(rs).hexdigest()!=r['process_samples_sha256']:raise ValueError('GPU replay identity')
            if sorted(w['seed'] for w in r['workers'])!=[42,43,44]:raise ValueError('GPU replay seeds')
            result['resource_replay']={'workers':{}}
            for w in r['workers']:
                values=[number(s['total_process_mib'][str(w['seed'])],1,81920) for s in json.loads(rs) if str(w['seed']) in s['total_process_mib']]
                if not values or w['epochs']!=120 or len(values)!=w['samples'] or max(values)!=w['sampled_total_process_peak_mib']:raise ValueError('GPU replay telemetry')
                result['resource_replay']['workers'][str(w['seed'])]={'gpu_index':number(w['gpu_index'],0,7),'samples':len(values),'sampled_total_process_peak_mib':max(values)}
        return result
    except (OSError,ValueError,KeyError,TypeError):return None

def structured_progress(repo, small=False):
    """Audited E33/E34 aggregates; never publish weights, priors or raw paths."""
    root=repo/'experiments'/('structured_policy_20260913_e34' if small else 'phase_budget_20260913_e33')
    try:
        if not (root/'dev/FROZEN.json').exists():return None
        raw=(root/'dev/FROZEN.json').read_bytes();cfg=json.loads(raw);h=hashlib.sha256(raw).hexdigest()
        if len(cfg['rows'])!=64 or cfg['arms']!=['stock','random','degree','original','untrained','cap42','cap43','cap44'] or cfg['seeds']!=[42,43,44]:raise ValueError('structured panel identity')
        result={'small_policy':small,'frozen_sha256':h,'instances':64,'trials':192,'target_cells':1536,'completed_cells':0,'phase':'preparing','observed':cfg['created'],'readout':None}
        if (root/'dev/STATUS.json').exists():
            s=json.loads((root/'dev/STATUS.json').read_text())
            if s['frozen_sha256']!=h or s['target_cells']!=1536 or type(s['completed_cells']) is not int or not 0<=s['completed_cells']<=1536:raise ValueError('structured progress')
            if s['phase'] not in ['running','terminal','failed']:raise ValueError('structured phase')
            result.update(phase='awaiting_audit' if s['phase']=='terminal' else s['phase'],completed_cells=s['completed_cells'],observed=s['observed'])
        if (root/'AUDIT.json').exists():
            audit=json.loads((root/'AUDIT.json').read_text());rr=(root/'dev/RESULTS.json').read_bytes();r=json.loads(rr)
            if audit['results_sha256']!=hashlib.sha256(rr).hexdigest() or audit['frozen_sha256']!=h or audit['verdict']!='integrity_pass_development_gates_failed' or audit['test_generated'] is not False:raise ValueError('structured audit')
            if r['cells']!=1536 or r['trials']!=192 or r['instances']!=64 or r['errors'] or r['engineering_pass'] or r['learning_pass'] or r['frozen_sha256']!=h:raise ValueError('structured readout')
            choice=json.loads((root/'SELECTION.json').read_text())
            if choice['engineering'] or choice['learning'] or choice['results_sha256']!=audit['results_sha256'] or (root/'test').exists():raise ValueError('structured stop')
            if result['completed_cells']!=1536 or result['phase']!='awaiting_audit':raise ValueError('structured incomplete stage')
            def summaries(values,limit=192):
                if set(values)!=set(cfg['arms']):raise ValueError('structured arms')
                out={}
                for a,v in values.items():
                    if any(type(v[k]) is not int or not 0<=v[k]<=limit for k in ['solved','sat','unsat','candidates']):raise ValueError('structured counts')
                    if not isinstance(v['par2_s'],(int,float)) or not math.isfinite(v['par2_s']) or not 0<=v['par2_s']<=2 or v['solved']!=v['sat']+v['unsat']:raise ValueError('structured time')
                    out[a]={k:v[k] for k in ['solved','sat','unsat','candidates','par2_s']}
                return out
            result.update(phase='terminal',readout={'sha256':audit['results_sha256'],'summaries':summaries(r['summaries']),
                'by_scale':{str(n):summaries(r['by_scale'][str(n)],48) for n in [24,32,48,64]},'no_test':True})
            for a,v in result['readout']['summaries'].items():
                parts=[s[a] for s in result['readout']['by_scale'].values()]
                if any(sum(p[k] for p in parts)!=v[k] for k in ['solved','sat','unsat','candidates']) or not math.isclose(sum(p['par2_s'] for p in parts)/4,v['par2_s'],abs_tol=1e-12):raise ValueError('structured scale reconciliation')
        return result
    except (OSError,ValueError,KeyError,TypeError):return None

def resident_progress(repo):
    """E35 conversion and fresh development stay separate; fixed public fields only."""
    root=repo/'experiments/resident_policy_20260913_e35'
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def number(v,maximum=2):
        if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=maximum:raise ValueError('resident numeric field')
        return v
    try:
        if not (root/'BENCHMARK_AUDIT.json').exists():return None
        a=json.loads((root/'BENCHMARK_AUDIT.json').read_text());b=json.loads((root/'BENCHMARK.json').read_text())
        if a['verdict']!='conversion_integrity_pass_not_learning_advantage' or a['benchmark_sha256']!=digest(root/'BENCHMARK.json') or a['frozen_sha256']!=digest(root/'BENCHMARK_FROZEN.json') or b['frozen_sha256']!=a['frozen_sha256'] or a['cells']!=480 or a['independent_formulas']!=8:raise ValueError('conversion audit')
        summaries={}
        for kind in ['old','cold','resident']:
            for mode in ['stock','tiny42']:
                key=kind+'_'+mode;v=b['summaries'][key]
                if v['count']!=80:raise ValueError('conversion count')
                summaries[key]={k:number(v[k],10) for k in ['mean','median','std','p95']}
        passed=all(summaries['resident_tiny42'][k]<=.8*summaries['old_tiny42'][k] for k in ['mean','median'])
        if a['conversion_pass'] is not passed or b['conversion_pass'] is not passed:raise ValueError('conversion gate')
        result={'benchmark':{'sha256':a['benchmark_sha256'],'conversion_pass':passed,'summaries':summaries,
            'instances':8,'repetitions':10,'cells':480,'resident_setup_s':number(b['resident_setup_s'],10)},'development':None}
        folder=root/'dev'
        if not (folder/'FROZEN.json').exists():return result
        cfg=json.loads((folder/'FROZEN.json').read_text());h=digest(folder/'FROZEN.json')
        arms=['stock','random','degree','static','untrained','tiny42','tiny43','tiny44']
        if len(cfg['rows'])!=64 or cfg['arms']!=arms or cfg['seeds']!=[42,43,44]:raise ValueError('resident design')
        state=json.loads((folder/'STATUS.json').read_text())
        if state['frozen_sha256']!=h or state['target_cells']!=1536 or type(state['completed_cells']) is not int or not 0<=state['completed_cells']<=1536 or state['phase'] not in ['running','failed','terminal']:raise ValueError('resident state')
        dev={'frozen_sha256':h,'phase':'awaiting_audit' if state['phase']=='terminal' else state['phase'],
             'completed_cells':state['completed_cells'],'target_cells':1536,'observed':state.get('observed',state['created']),'readout':None}
        result['development']=dev
        if not (folder/'AUDIT.json').exists():return result
        audit=json.loads((folder/'AUDIT.json').read_text());r=json.loads((folder/'RESULTS.json').read_text());choice=json.loads((root/'SELECTION.json').read_text())
        if audit['verdict']!='integrity_pass' or audit['frozen_sha256']!=h or audit['results_sha256']!=digest(folder/'RESULTS.json') or r['frozen_sha256']!=h or audit['selection_sha256']!=digest(root/'SELECTION.json') or choice['results_sha256']!=audit['results_sha256']:raise ValueError('resident readout identity')
        if audit['engineering'] is not False or audit['learning'] is not False or r['engineering_pass'] is not False or r['learning_pass'] is not False or choice['engineering'] is not False or choice['learning'] is not False or audit['test_generated'] is not False or (root/'test').exists():raise ValueError('resident negative stop')
        if dev['phase']!='awaiting_audit' or dev['completed_cells']!=1536 or r['cells']!=1536 or r['trials']!=192 or r['instances']!=64:raise ValueError('resident completeness')
        def summary(values,limit):
            if set(values)!=set(arms):raise ValueError('resident arms')
            out={}
            for arm,v in values.items():
                if any(type(v[k]) is not int or not 0<=v[k]<=limit for k in ['solved','sat','unsat','candidates']) or v['solved']!=v['sat']+v['unsat']:raise ValueError('resident counts')
                out[arm]={k:v[k] for k in ['solved','sat','unsat','candidates']};out[arm]['par2_s']=number(v['par2_s'])
                out[arm]['median_s']=number(v['full_s']['median'])
            return out
        sm=summary(r['summaries'],192);scales={str(n):summary(r['by_scale'][str(n)],48) for n in [24,32,48,64]}
        for arm,v in sm.items():
            if any(sum(s[arm][k] for s in scales.values())!=v[k] for k in ['solved','sat','unsat','candidates']) or not math.isclose(sum(s[arm]['par2_s'] for s in scales.values())/4,v['par2_s'],abs_tol=1e-12):raise ValueError('resident scale reconciliation')
        dev.update(phase='terminal',readout={'sha256':audit['results_sha256'],'summaries':sm,'by_scale':scales,'no_test':True})
        return result
    except (OSError,ValueError,KeyError,TypeError):return None

def decoder_utility_progress(repo):
    """E36 audited validation-only evidence; never publish graphs, choices or paths."""
    root=repo/'experiments/decoder_utility_20260913_e36'
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def num(v,limit=64,integer=False):
        if type(v) not in (int,float) or (integer and type(v) is not int) or not math.isfinite(v) or not 0<=v<=limit:raise ValueError('utility numeric bound')
        return v
    try:
        a=json.loads((root/'AUDIT.json').read_text());r=json.loads((root/'READOUT.json').read_text())
        h=digest(root/'READOUT.json');f=digest(root/'READOUT_FROZEN.json')
        if a['verdict']!='integrity_pass' or a['readout_sha256']!=h or a['frozen_sha256']!=f or r['frozen_sha256']!=f or r['scope']!='selected_validation_not_solver_performance':raise ValueError('utility audit identity')
        if a['cnfs']!=320 or a['diagnostic_rows']!=160 or a['cpu_checkpoint_replays']!=192:raise ValueError('utility audit count')
        if any(x[k] is not False for x in [a,r] for k in ['dev_generated','test_generated']) or (root/'dev').exists() or (root/'test').exists():raise ValueError('utility stop')
        teacher={}
        for split,count in [('train',256),('validation',64)]:
            s=r['teacher'][split]
            if s['count']!=count:raise ValueError('utility split')
            teacher[split]={'count':count,**{k:num(s[k],count,True) for k in ['variable_graphs','degree_covers','oracle_covers']},
                            **{k:num(s[k]) for k in ['degree_mean','oracle_mean']}}
            if s['oracle_covers']<s['degree_covers'] or s['oracle_mean']<s['degree_mean']:raise ValueError('utility oracle')
        v=teacher['validation'];label_gate=v['variable_graphs']>=16 and (v['oracle_covers']-v['degree_covers']>=2 or v['oracle_mean']-v['degree_mean']>=.10)
        workers={}
        if set(r['workers'])!={'42','43','44'}:raise ValueError('utility seeds')
        for seed in ['42','43','44']:
            w=r['workers'][seed]
            if w['steps']!=600 or w['seed']!=int(seed):raise ValueError('utility worker')
            out={'steps':600,**{k:num(w[k],limit,True) for k,limit in [('selected_step',600),('covers',64),('changed_choices',64),('samples',10000),('process_peak_mib',8192)]},
                 'mean_q':num(w['mean_q']),'elapsed_s':num(w['elapsed_s'],300),'by_scale':{}}
            if not out['samples']:raise ValueError('utility telemetry')
            for n in ['24','32','48','64']:
                s=w['by_scale'][n]
                if s['count']!=16:raise ValueError('utility scale count')
                out['by_scale'][n]={'count':16,**{k:num(s[k],16,True) for k in ['baseline_covers','covers']},
                                    **{k:num(s[k]) for k in ['baseline_mean','mean_q']}}
            scales=out['by_scale'].values()
            if sum(s['covers'] for s in scales)!=out['covers'] or not math.isclose(sum(s['mean_q'] for s in scales)/4,out['mean_q'],abs_tol=1e-12):raise ValueError('utility scale total')
            if sum(s['baseline_covers'] for s in scales)!=v['degree_covers'] or not math.isclose(sum(s['baseline_mean'] for s in scales)/4,v['degree_mean'],abs_tol=1e-12):raise ValueError('utility baseline total')
            workers[seed]=out
        training_gate=all(w['mean_q']>v['degree_mean'] and w['covers']>=v['degree_covers'] for w in workers.values())
        if training_gate or not label_gate or any(x['training_gate'] is not training_gate or x['label_gate'] is not label_gate for x in [a,r]):raise ValueError('utility gates')
        diagnostic={}
        for arm in ['untrained','tiny42','tiny43','tiny44']:
            s=r['diagnostic']['validation'][arm];diagnostic[arm]={k:num(s[k],32,True) for k in ['greedy_covers','deployed_covers']}
        return dict(sha256=h,phase='terminal',scope=r['scope'],diagnostic=diagnostic,teacher=teacher,workers=workers,
                    label_gate=label_gate,training_gate=training_gate,no_dev=True,no_test=True,training_elapsed_s=num(r['training_elapsed_s'],360))
    except (OSError,ValueError,KeyError,TypeError):return None

def stability_progress(repo):
    """E37 validation gate is not solver performance; export only checked aggregates."""
    root=repo/'experiments/stability_20260913_e37';candidates=['ce_mlp','pair_mlp','ce_graph','pair_graph']
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def num(v,limit=64,integer=False):
        if type(v) not in (int,float) or (integer and type(v) is not int) or not math.isfinite(v) or not 0<=v<=limit:raise ValueError('stability numeric bound')
        return v
    try:
        r=json.loads((root/'READOUT.json').read_text());a=json.loads((root/'AUDIT.json').read_text());h=digest(root/'READOUT.json');f=digest(root/'READOUT_FROZEN.json')
        if a['verdict']!='integrity_pass' or a['readout_sha256']!=h or a['frozen_sha256']!=f or r['frozen_sha256']!=f or r['scope']!='factorial_selected_validation_not_solver_performance':raise ValueError('stability audit')
        if a['cnfs']!=1280 or a['reference_choices']!=53760 or a['cpu_checkpoint_choices']!=3072 or a['validation_history_rechecks']!=73728:raise ValueError('stability audit counts')
        if any(x[k] is not False for x in [a,r] for k in ['dev_generated','test_generated']) or (root/'dev').exists() or (root/'test').exists():raise ValueError('stability stage')
        def base(s,count):
            if s['count']!=count:raise ValueError('stability denominator')
            return dict(count=count,**{k:num(s[k],count,True) for k in ['baseline_covers','oracle_covers','variable_graphs']},
                        **{k:num(s[k]) for k in ['baseline_mean','oracle_mean']})
        baseline=dict(summary=base(r['baseline']['summary'],256),by_scale={str(n):base(r['baseline']['by_scale'][str(n)],64) for n in [24,32,48,64]})
        b=baseline['summary']
        if any(sum(s[k] for s in baseline['by_scale'].values())!=b[k] for k in ['baseline_covers','oracle_covers','variable_graphs']):raise ValueError('stability baseline sums')
        if any(not math.isclose(sum(s[k] for s in baseline['by_scale'].values())/4,b[k],abs_tol=1e-12) for k in ['baseline_mean','oracle_mean']):raise ValueError('stability baseline means')
        results={}
        if set(r['results'])!=set(candidates):raise ValueError('stability candidates')
        for c in candidates:
            raw=r['results'][c];workers={};params=241 if c.endswith('mlp') else 4641
            if raw['parameters']!=params or set(raw['workers'])!={'42','43','44'}:raise ValueError('stability structure')
            for seed in ['42','43','44']:
                w=raw['workers'][seed]
                if w['steps']!=600 or w['count']!=256 or w['parameters']!=params:raise ValueError('stability training complete')
                out=dict(steps=600,count=256,mean_q=num(w['mean_q']),covers=num(w['covers'],256,True),selected_step=num(w['selected_step'],600,True),by_scale={})
                for n in ['24','32','48','64']:
                    s=w['by_scale'][n]
                    if s['count']!=64:raise ValueError('stability scale')
                    out['by_scale'][n]=dict(count=64,mean_q=num(s['mean_q']),covers=num(s['covers'],64,True))
                if sum(s['covers'] for s in out['by_scale'].values())!=out['covers'] or not math.isclose(sum(s['mean_q'] for s in out['by_scale'].values())/4,out['mean_q'],abs_tol=1e-12):raise ValueError('stability scale arithmetic')
                passed=out['mean_q']>=b['baseline_mean']+.02 and out['covers']>=b['baseline_covers']+2 and all(s['mean_q']>=baseline['by_scale'][n]['baseline_mean'] and s['covers']>=baseline['by_scale'][n]['baseline_covers'] for n,s in out['by_scale'].items())
                if w['passed'] is not passed or w['cpu_choice_disagreements']!=0 or w['cpu_utility_disagreements']!=0:raise ValueError('stability seed gate/parity')
                out['passed']=passed;workers[seed]=out
            passed=all(w['passed'] for w in workers.values());score=min(w['mean_q']-b['baseline_mean'] for w in workers.values())-.001*math.log2(params/241)
            if raw['passed'] is not passed or not math.isclose(raw['selection_score'],score,abs_tol=1e-12):raise ValueError('stability candidate gate')
            results[c]=dict(parameters=params,workers=workers,passed=passed,selection_score=score)
        passing=[c for c in candidates if results[c]['passed']];selected=max(passing,key=lambda c:results[c]['selection_score']) if passing else None
        if r['selected']!=selected or a['selected']!=selected:raise ValueError('stability selection')
        predictions=dict(H1=all(results['pair_mlp']['workers'][s]['mean_q']-results['ce_mlp']['workers'][s]['mean_q']>=.02 for s in ['42','43','44']),
            H2=all(results[loss+'_graph']['workers'][s]['mean_q']-results[loss+'_mlp']['workers'][s]['mean_q']>=.02 for s in ['42','43','44'] for loss in ['ce','pair']))
        if r['predictions']!=predictions:raise ValueError('stability predictions')
        resources={s:dict(samples=num(r['resources'][s]['samples'],10000,True),process_peak_mib=num(r['resources'][s]['process_peak_mib'],8192,True)) for s in ['42','43','44']}
        if any(not x['samples'] for x in resources.values()):raise ValueError('stability resource samples')
        return dict(sha256=h,baseline=baseline,results=results,selected=selected,predictions=predictions,resources=resources,
            training_elapsed_s=num(r['training_elapsed_s'],1260),diagnostic_tie_fraction=num(r['diagnostic']['initial']['tie_gradient_fraction'],1),
            no_dev=True,no_test=True,scope=r['scope'],phase='terminal')
    except (OSError,ValueError,KeyError,TypeError):return None

def paired_launch_progress(repo):
    """E60 is a paired engineering replay, never a learned-policy win."""
    folder = repo / 'experiments/native_first_20260915_e60/run'
    try:
        raw = (folder / 'RESULTS.json').read_bytes()
        r = json.loads(raw)
        audit = json.loads((folder / 'AUDIT.json').read_bytes())
        fh = hashlib.sha256((folder / 'FROZEN.json').read_bytes()).hexdigest()
        digest = hashlib.sha256(raw).hexdigest()
        if audit.get('verdict') != 'integrity_pass' or audit.get('results_sha256') != digest:
            return None
        if r.get('frozen_sha256') != fh or audit.get('frozen_sha256') != fh:
            return None
        if (r.get('cells'), r.get('graphs'), r.get('trials_per_arm'), r.get('repeats')) != (3456,128,1152,3):
            return None
        if audit.get('cells') != 3456 or audit.get('checked_masks') != 3456:
            return None
        if r.get('output_mismatches') != 0 or audit.get('output_mismatches') != 0:
            return None
        if r.get('independent_confirmation') is not False or r.get('learned_policy_advantage') is not False:
            return None
        arms = ['taskset','inherited','unbound']
        if set(r['summaries']) != set(arms):
            return None
        def finite(x):
            if type(x) not in (int,float) or not math.isfinite(x) or x < 0:
                raise ValueError('invalid launch cost')
            return x
        regimes = {}
        for a in arms:
            s = r['summaries'][a]
            if (s['count'],s['solved'],s['errors']) != (1152,1152,0):
                return None
            regimes[a] = dict(par2_ms=finite(s['par2_ms']),median_ms=finite(s['raw_s']['median']),
                setup_mean_ms=finite(s['setup_s']['mean']),native_mean_ms=finite(s['native_wall_ms']['mean']),
                audited_total_ms=finite(s['audited_total_s']['mean']),solved=s['solved'])
        c = r['contrasts']['inherited_vs_taskset']
        comparison = {k: finite(c[k]) for k in ['gain_ms','gain_pct','simultaneous95_lower_ms','simultaneous95_upper_ms']}
        if not comparison['simultaneous95_lower_ms'] <= comparison['gain_ms'] <= comparison['simultaneous95_upper_ms']:
            return None
        if set(r['by_scale']) != {'24','32','48','64'}:
            return None
        scales = {k: {a: finite(v[a]['par2_ms']) for a in arms} for k,v in r['by_scale'].items()}
        return dict(sha256=digest,phase='terminal',graphs=128,cells=3456,trials_per_arm=1152,
                    regimes=regimes,comparison=comparison,by_scale=scales,output_mismatches=0,
                    scope='同图同核配对工程复测；非独立确认、非模型优势')
    except (OSError,ValueError,KeyError,TypeError):
        return None


def cache_affinity_v2_progress(repo):
    """Allowlist an audited independent cache replication, including negative gates."""
    folder=repo/'experiments/cache_affinity_20260915_v2/dev'
    try:
        raw=(folder/'RESULTS.json').read_bytes();r=json.loads(raw)
        a=json.loads((folder/'AUDIT.json').read_text())
        fh=hashlib.sha256((folder/'FROZEN.json').read_bytes()).hexdigest()
        h=hashlib.sha256(raw).hexdigest()
        if a.get('verdict')!='integrity_pass' or a.get('results_sha256')!=h:
            return None
        if r.get('frozen_sha256')!=fh or a.get('frozen_sha256')!=fh:
            return None
        if (r.get('cells'),r.get('independent_graphs'),r.get('repeats'))!=(4608,128,3):
            return None
        if (a.get('cells'),a.get('graphs'),a.get('reference_agreements'),a.get('cache_hits'))!=(4608,128,4608,768):
            return None
        if r.get('learned_advantage') is not False or r.get('source_integrity') is not True:
            return None
        def number(x,negative=False):
            if type(x) not in (int,float) or not math.isfinite(x) or abs(x)>100000 or (x<0 and not negative):
                raise ValueError('cache affinity numeric field')
            return x
        def summary(s,cells,hits):
            if s['cells']!=cells or s['cache_hits']!=hits:
                raise ValueError('cache affinity denominators/hits')
            for k in ('solved','timeouts'):
                if type(s[k]) is not int or not 0<=s[k]<=cells:
                    raise ValueError('cache affinity counts')
            return dict(par2_ms=number(s['par2_ms']),solved=s['solved'],timeouts=s['timeouts'],cells=cells,cache_hits=hits)
        arms=('random32','cache_random32');regimes=('cold','resident')
        if set(r['summaries'])!=set(regimes) or set(r['by_scale'])!=set(regimes):
            return None
        sm={};scales={};ci={};gates={}
        for reg in regimes:
            if set(r['summaries'][reg])!=set(arms) or set(r['by_scale'][reg])!={'24','32','48','64'}:
                return None
            sm[reg]={arm:summary(r['summaries'][reg][arm],1152,768 if reg=='resident' and arm=='cache_random32' else 0) for arm in arms}
            scales[reg]={n:{arm:summary(v[arm],288,192 if reg=='resident' and arm=='cache_random32' else 0) for arm in arms} for n,v in r['by_scale'][reg].items()}
            for arm in arms:
                if sum(v[arm]['solved'] for v in scales[reg].values())!=sm[reg][arm]['solved']:
                    return None
                if not math.isclose(statistics.mean(v[arm]['par2_ms'] for v in scales[reg].values()),sm[reg][arm]['par2_ms'],abs_tol=1e-9):
                    return None
            c=r['contrasts'][reg]
            if c['independent_graphs']!=128:
                return None
            ci[reg]={k:number(c[k],True) for k in ('gain_ms','lower_ms','upper_ms')}
            base,cached=(sm[reg][arm] for arm in arms)
            if not math.isclose(ci[reg]['gain_ms'],base['par2_ms']-cached['par2_ms'],abs_tol=1e-9) or ci[reg]['lower_ms']>ci[reg]['upper_ms']:
                return None
            gates[reg]=bool(cached['par2_ms']<=.95*base['par2_ms'] and cached['solved']>=base['solved'] and ci[reg]['lower_ms']>0 and all(
                v['cache_random32']['par2_ms']<=v['random32']['par2_ms'] and v['cache_random32']['solved']>=v['random32']['solved'] for v in scales[reg].values()))
        if r['gates']!=gates or a['gates']!=gates or r['all_lifecycles_pass'] is not all(gates.values()):
            return None
        return dict(sha256=h,frozen_sha256=fh,phase='terminal',cells=4608,graphs=128,repeats=3,
                    regimes=sm,by_scale=scales,contrasts=ci,gates=gates,learning_pass=False,
                    scope='Fresh graphs; repeated-CNF infrastructure only; not learned-policy confirmation')
    except (OSError,ValueError,KeyError,TypeError):
        return None


def native_first_progress(repo):
    """E38 native/full-cost aggregate; never export raw requests, paths or PIDs."""
    root=repo/'experiments/native_first_20260913_e38';dev=root/'dev'
    try:
        if not dev.exists() or (root/'test').exists(): return None
        r=json.loads((dev/'RESULTS.json').read_text());cfg=json.loads((dev/'FROZEN.json').read_text());st=json.loads((dev/'STATUS.json').read_text())
        h=hashlib.sha256((dev/'RESULTS.json').read_bytes()).hexdigest();fh=hashlib.sha256((dev/'FROZEN.json').read_bytes()).hexdigest()
        if st['phase']!='terminal' or st['completed_cells']!=7680 or r['cells']!=7680 or r['trials']!=384 or r['frozen_sha256']!=fh:raise ValueError('native E38 identity')
        if set(r['summaries'])!={'cold','resident'} or r['learning_pass'] is not False or r['Hcheap_prediction_pass'] is not True:raise ValueError('native E38 gate')
        def n(v,limit=100000):
            if type(v) not in (int,float) or isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=limit:raise ValueError('native E38 numeric')
            return v
        arms=['stock','degree32','random32','pair42','pair43','pair44'];out={}
        for regime in ['cold','resident']:
            out[regime]={}
            for arm in arms:
                v=r['summaries'][regime][arm];out[regime][arm]=dict(par2_ms=n(v['par2_s']*1000),solved=n(v['solved'],384),sat=n(v['sat'],384),unsat=n(v['unsat'],384),timeouts=n(v['timeouts'],384))
        scale={}
        for regime in ['cold','resident']:
            scale[regime]={}
            for size,vv in r['by_scale'][regime].items():
                scale[regime][size]={a:dict(par2_ms=n(vv[a]['par2_s']*1000),solved=n(vv[a]['solved'],96)) for a in arms}
        result=dict(sha256=h,frozen_sha256=fh,phase='terminal',instances=128,trials=384,cells=7680,target_cells=7680,
                    regimes=out,by_scale=scale,learning_pass=False,cheap_prediction=True,compute_median_gain_pct=n(json.loads((root/'COMPAT.json').read_text())['core_median_gain']*100,100),scope='E38 native full-cost development; validation-only model selection excluded')
        prior_digest={}
        for e in range(38,46):
            fp=repo/f'experiments/native_first_20260913_e{e}/dev/FROZEN.json'
            if fp.exists():
                prior_digest[e]={x['sha256'] for x in json.loads(fp.read_text()).get('rows',[])}
        if all(e in prior_digest for e in range(38,46)) and all(prior_digest[e]==prior_digest[38] for e in range(39,46)):
            result['cohort_correction']='E39–E45 reused the E38 128-instance cohort; these are repeated measurements, not independent samples.'
        follow=repo/'experiments/native_first_20260913_e39/dev/RESULTS.json'
        if follow.exists() and not (repo/'experiments/native_first_20260913_e39/test').exists():
            fr=json.loads(follow.read_text());
            if fr.get('cells')!=6144 or fr.get('trials')!=384 or fr.get('learning_pass') is not False:raise ValueError('native E39 gate')
            arms=['stock','random32','adaptive','pair42','pair43','pair44'];fout={}
            for regime in ['cold','resident']:
                fout[regime]={a:dict(par2_ms=n(fr['summaries'][regime][a]['par2_s']*1000),solved=n(fr['summaries'][regime][a]['solved'],384)) for a in arms}
            result['followup']=dict(sha256=hashlib.sha256(follow.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,
                                    regimes=fout,learning_pass=False,cheap_prediction=True,scope='E39 scale-adaptive development; no confirmation')
        online=repo/'experiments/native_first_20260913_e40/dev/RESULTS.json'
        if online.exists() and not (repo/'experiments/native_first_20260913_e40/test').exists():
            oraw=json.loads(online.read_text())
            if oraw.get('cells')!=6144 or oraw.get('trials')!=384 or oraw.get('learning_pass') is not False:raise ValueError('native E40 gate')
            arms=['stock','degree32','random32','online'];oout={}
            for regime in ['cold','resident']:
                oout[regime]={a:dict(par2_ms=n(oraw['summaries'][regime][a]['par2_s']*1000),solved=n(oraw['summaries'][regime][a]['solved'],384)) for a in arms}
            result['online_followup']=dict(sha256=hashlib.sha256(online.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,
                                           regimes=oout,learning_pass=False,cheap_prediction=True,scope='E40 instance-level online probe development; probe cost charged; no confirmation')
        gate=repo/'experiments/native_first_20260913_e41/dev/RESULTS.json'
        if gate.exists() and not (repo/'experiments/native_first_20260913_e41/test').exists():
            graw=json.loads(gate.read_text())
            if graw.get('cells')!=6144 or graw.get('trials')!=384 or graw.get('learning_pass') is not False:raise ValueError('native E41 gate')
            arms=['stock','degree32','random32','gate'];gout={}
            for regime in ['cold','resident']:
                gout[regime]={a:dict(par2_ms=n(graw['summaries'][regime][a]['par2_s']*1000),solved=n(graw['summaries'][regime][a]['solved'],384)) for a in arms}
            result['gate_followup']=dict(sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,
                                         regimes=gout,learning_pass=False,cheap_prediction=True,scope='E41 zero-extra-process feature gate; no confirmation')
        for key, folder_name, arm_name, scope_name in [('phase_followup','native_first_20260913_e42','phase42','E42 solver phase-hint development'),('one_followup','native_first_20260913_e43','phase42one','E43 single-phase-hint development')]:
            p=repo/f'experiments/{folder_name}/dev/RESULTS.json'
            if p.exists() and not (repo/f'experiments/{folder_name}/test').exists():
                rr=json.loads(p.read_text())
                if rr.get('cells')!=6144 or rr.get('trials')!=384 or rr.get('learning_pass') is not False:raise ValueError('native followup gate')
                vals={}
                for regime in ['cold','resident']:
                    vals[regime]={a:dict(par2_ms=n(rr['summaries'][regime][a]['par2_s']*1000),solved=n(rr['summaries'][regime][a]['solved'],384)) for a in ['stock','degree32','random32',arm_name]}
                result[key]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=vals,learning_pass=False,cheap_prediction=True,scope=scope_name+'; no confirmation')
        tail=repo/'experiments/native_first_20260913_e44/dev/RESULTS.json'
        if tail.exists() and not (repo/'experiments/native_first_20260913_e44/test').exists():
            tr=json.loads(tail.read_text())
            if tr.get('cells')!=6144 or tr.get('trials')!=384 or tr.get('learning_pass') is not False:raise ValueError('native E44 gate')
            tv={}
            for regime in ['cold','resident']:
                tv[regime]={a:dict(par2_ms=n(tr['summaries'][regime][a]['par2_s']*1000),solved=n(tr['summaries'][regime][a]['solved'],384)) for a in ['stock','degree32','random32','tail']}
            result['tail_followup']=dict(sha256=hashlib.sha256(tail.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=tv,learning_pass=False,cheap_prediction=True,scope='E44 solver restart-tail development; no confirmation')
        conditional=repo/'experiments/native_first_20260913_e45/dev/RESULTS.json'
        if conditional.exists() and not (repo/'experiments/native_first_20260913_e45/test').exists():
            cr=json.loads(conditional.read_text())
            if cr.get('cells')!=6144 or cr.get('trials')!=384 or cr.get('learning_pass') is not False:raise ValueError('native E45 gate')
            cv={}
            for regime in ['cold','resident']:
                cv[regime]={a:dict(par2_ms=n(cr['summaries'][regime][a]['par2_s']*1000),solved=n(cr['summaries'][regime][a]['solved'],384)) for a in ['stock','degree32','random32','conditional_tail']}
            result['conditional_followup']=dict(sha256=hashlib.sha256(conditional.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=cv,learning_pass=False,cheap_prediction=True,scope='E45 conditional solver-tail development; no confirmation')
        runtime=repo/'experiments/native_first_20260913_e46/dev/RESULTS.json'
        if runtime.exists() and not (repo/'experiments/native_first_20260913_e46/test').exists():
            rr=json.loads(runtime.read_text())
            if rr.get('cells')!=6144 or rr.get('trials')!=384 or rr.get('learning_pass') is not False:raise ValueError('native E46 gate')
            rv={}
            for regime in ['cold','resident']:
                rv[regime]={a:dict(par2_ms=n(rr['summaries'][regime][a]['par2_s']*1000),solved=n(rr['summaries'][regime][a]['solved'],384)) for a in ['stock','degree32','random32','runtime_tail']}
            result['runtime_followup']=dict(sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=rv,learning_pass=False,cheap_prediction=True,scope='E46 deduplicated runtime-tail development; no confirmation')
        wide=repo/'experiments/native_first_20260913_e47/dev/RESULTS.json'
        if wide.exists() and not (repo/'experiments/native_first_20260913_e47/test').exists():
            wr=json.loads(wide.read_text())
            if wr.get('cells')!=6144 or wr.get('trials')!=384 or wr.get('learning_pass') is not False:raise ValueError('native E47 gate')
            wv={}
            for regime in ['cold','resident']:
                wv[regime]={a:dict(par2_ms=n(wr['summaries'][regime][a]['par2_s']*1000),solved=n(wr['summaries'][regime][a]['solved'],384)) for a in ['stock','degree32','random32','pair42wide']}
            result['wide_followup']=dict(sha256=hashlib.sha256(wide.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=wv,learning_pass=False,cheap_prediction=True,scope='E47 equal-budget learned ranking development; no confirmation')
        portfolio=repo/'experiments/native_first_20260913_e48/dev/RESULTS.json'
        if portfolio.exists() and not (repo/'experiments/native_first_20260913_e48/test').exists():
            pr=json.loads(portfolio.read_text())
            if pr.get('cells')!=6144 or pr.get('trials')!=384 or pr.get('learning_pass') is not False:raise ValueError('native E48 gate')
            pv={}
            for regime in ['cold','resident']:
                pv[regime]={a:dict(par2_ms=n(pr['summaries'][regime][a]['par2_s']*1000),solved=n(pr['summaries'][regime][a]['solved'],384)) for a in ['stock','degree32','random32','portfolio']}
            result['portfolio_followup']=dict(sha256=hashlib.sha256(portfolio.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=pv,learning_pass=False,cheap_prediction=True,scope='E48 lifecycle-aware portfolio development; no confirmation')
        cache=repo/'experiments/native_first_20260913_e49/dev/RESULTS.json'
        if cache.exists() and not (repo/'experiments/native_first_20260913_e49/test').exists():
            xr=json.loads(cache.read_text())
            if xr.get('cells')!=6912 or xr.get('trials')!=384 or xr.get('learning_pass') is not False:raise ValueError('native E49 gate')
            xv={}
            for regime in ['cold','resident']:
                xv[regime]={a:dict(par2_ms=n(xr['summaries'][regime][a]['par2_s']*1000),solved=n(xr['summaries'][regime][a]['solved'],384)) for a in ['random32','cache_random32','cache_pair42']}
            result['cache_followup']=dict(sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6912,regimes=xv,learning_pass=False,cheap_prediction=True,scope='E49 same-process CNF parse-cache development; no confirmation')
        gate50=repo/'experiments/native_first_20260913_e50/dev/RESULTS.json'
        if gate50.exists() and not (repo/'experiments/native_first_20260913_e50/test').exists():
            xr=json.loads(gate50.read_text())
            if xr.get('cells')!=6144 or xr.get('trials')!=384 or xr.get('learning_pass') is not False:raise ValueError('native E50 gate')
            xv={}
            for regime in ['cold','resident']:
                xv[regime]={a:dict(par2_ms=n(xr['summaries'][regime][a]['par2_s']*1000),solved=n(xr['summaries'][regime][a]['solved'],384)) for a in ['random32','runtime_gate','pair42']}
            result['runtime_gate_followup']=dict(sha256=hashlib.sha256(gate50.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=xv,learning_pass=False,cheap_prediction=True,scope='E50 runtime conflict probe gate development; no confirmation')
        pig=repo/'experiments/native_first_20260913_e51/dev/RESULTS.json'
        if pig.exists() and not (repo/'experiments/native_first_20260913_e51/test').exists():
            xr=json.loads(pig.read_text())
            if xr.get('cells')!=6144 or xr.get('trials')!=384 or xr.get('learning_pass') is not False:raise ValueError('native E51 gate')
            xv={}
            for regime in ['cold','resident']:
                xv[regime]={a:dict(par2_ms=n(xr['summaries'][regime][a]['par2_s']*1000),solved=n(xr['summaries'][regime][a]['solved'],384),candidates=n(xr['summaries'][regime][a]['candidates'],384)) for a in ['random32','piggyback_pair','pair42']}
            result['piggyback_followup']=dict(sha256=hashlib.sha256(pig.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=xv,learning_pass=False,cheap_prediction=True,scope='E51 piggyback runtime fallback development; no confirmation')
        lazy=repo/'experiments/native_first_20260913_e52/dev/RESULTS.json'
        if lazy.exists() and not (repo/'experiments/native_first_20260913_e52/test').exists():
            xr=json.loads(lazy.read_text())
            if xr.get('cells')!=6144 or xr.get('trials')!=384 or xr.get('learning_pass') is not False:raise ValueError('native E52 gate')
            xv={}
            for regime in ['cold','resident']:
                xv[regime]={a:dict(par2_ms=n(xr['summaries'][regime][a]['par2_s']*1000),solved=n(xr['summaries'][regime][a]['solved'],384),candidates=n(xr['summaries'][regime][a]['candidates'],384)) for a in ['random32','piggyback_pair','pair42']}
            result['lazy_followup']=dict(sha256=hashlib.sha256(lazy.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=xv,learning_pass=False,cheap_prediction=True,scope='E52 lazy piggyback runtime fallback development; no confirmation')
        one=repo/'experiments/native_first_20260913_e53/dev/RESULTS.json'
        if one.exists() and not (repo/'experiments/native_first_20260913_e53/test').exists():
            xr=json.loads(one.read_text())
            if xr.get('cells')!=6144 or xr.get('trials')!=384 or xr.get('learning_pass') is not False:raise ValueError('native E53 gate')
            xv={}
            for regime in ['cold','resident']:
                xv[regime]={a:dict(par2_ms=n(xr['summaries'][regime][a]['par2_s']*1000),solved=n(xr['summaries'][regime][a]['solved'],384),candidates=n(xr['summaries'][regime][a]['candidates'],384)) for a in ['random32','piggyback_pair','pair42']}
            result['oneshot_followup']=dict(sha256=hashlib.sha256(one.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=xv,learning_pass=False,cheap_prediction=True,scope='E53 one-shot lazy piggyback development; no confirmation')
        corr=repo/'experiments/native_first_20260913_e54/dev/RESULTS.json'
        if corr.exists() and not (repo/'experiments/native_first_20260913_e54/test').exists():
            xr=json.loads(corr.read_text());
            if xr.get('cells')!=6144 or xr.get('trials')!=384 or xr.get('learning_pass') is not False:raise ValueError('native E54 gate')
            xv={regime:{a:dict(par2_ms=n(xr['summaries'][regime][a]['par2_s']*1000),solved=n(xr['summaries'][regime][a]['solved'],384),candidates=n(xr['summaries'][regime][a]['candidates'],384)) for a in ['random32','piggyback_pair']} for regime in ['cold','resident']}
            result['corrected_followup']=dict(sha256=hashlib.sha256(corr.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=6144,regimes=xv,learning_pass=False,cheap_prediction=True,scope='E54 corrected first-pass budget; fallback telemetry serialized too early to establish execution count')
        affinity=repo/'experiments/native_first_20260915_e55/dev/RESULTS.json'
        if affinity.exists() and not (repo/'experiments/native_first_20260915_e55/test').exists():
            xr=json.loads(affinity.read_text())
            if xr.get('cells')!=2304 or xr.get('trials')!=384 or xr.get('learning_pass') is not False:raise ValueError('native E55 affinity gate')
            xv={regime:{a:dict(par2_ms=n(xr['summaries'][regime][a]['par2_s']*1000),solved=n(xr['summaries'][regime][a]['solved'],384),candidates=n(xr['summaries'][regime][a]['candidates'],384),parse_median_ms=n(xr['summaries'][regime][a]['components']['parse_s']['median']*1000)) for a in ['random32','cache_random32','cache_pair42']} for regime in ['cold','resident']}
            result['affinity_followup']=dict(sha256=hashlib.sha256(affinity.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=2304,regimes=xv,learning_pass=False,cheap_prediction=True,resident_repeat_reuse_requests=256,scope='E55 graph-to-worker affinity factorial; parse cache confirmed but learned ranking gate failed')
        cache_only=repo/'experiments/native_first_20260915_e56/dev/RESULTS.json'
        if cache_only.exists() and not (repo/'experiments/native_first_20260915_e56/test').exists():
            xr=json.loads(cache_only.read_text())
            if xr.get('cells')!=1536 or xr.get('trials')!=384 or xr.get('learning_pass') is not False:raise ValueError('native E56 gate')
            xv={regime:{a:dict(par2_ms=n(xr['summaries'][regime][a]['par2_s']*1000),solved=n(xr['summaries'][regime][a]['solved'],384),candidates=n(xr['summaries'][regime][a]['candidates'],384),parse_median_ms=n(xr['summaries'][regime][a]['components']['parse_s']['median']*1000)) for a in ['random32','cache_random32']} for regime in ['cold','resident']}
            result['cache_only_followup']=dict(sha256=hashlib.sha256(cache_only.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=1536,regimes=xv,learning_pass=False,resident_gate=bool(xr['gates']['resident']['cache_random32']),cold_gate=bool(xr['gates']['cold']['cache_random32']),resident_repeat_reuse_requests=256,scope='E56 independent cache-only replication; resident gate passes, cold lifecycle gate fails')
        pool=repo/'experiments/native_first_20260915_e57_pool_break_even.json'
        if pool.exists():
            xr=json.loads(pool.read_text());
            if set(xr.get('arms',{}))!={'random32','cache_random32'} or not {'3','9','27','81'}<=set(xr['arms']['random32']):raise ValueError('native E57 pool diagnostic')
            result['pool_break_even']=dict(sha256=hashlib.sha256(pool.read_bytes()).hexdigest(),scope='E57 E56-row persistent-worker amortization diagnostic; not independent confirmation',batch_sizes=[3,9,27,81],random32={k:n(v['charged_mean_ms']) for k,v in xr['arms']['random32'].items()},cache_random32={k:n(v['charged_mean_ms']) for k,v in xr['arms']['cache_random32'].items()})
        affinity_fix=repo/'experiments/native_first_20260915_e59/dev/RESULTS.json'
        if affinity_fix.exists() and not (repo/'experiments/native_first_20260915_e59/test').exists():
            xr=json.loads(affinity_fix.read_text())
            if xr.get('cells')!=1536 or xr.get('trials')!=384 or xr.get('learning_pass') is not False:raise ValueError('native E59 gate')
            xv={regime:{a:dict(par2_ms=n(xr['summaries'][regime][a]['par2_s']*1000),solved=n(xr['summaries'][regime][a]['solved'],384),candidates=n(xr['summaries'][regime][a]['candidates'],384),parse_median_ms=n(xr['summaries'][regime][a]['components']['parse_s']['median']*1000)) for a in ['random32','cache_random32']} for regime in ['cold','resident']}
            result['affinity_fix_followup']=dict(sha256=hashlib.sha256(affinity_fix.read_bytes()).hexdigest(),phase='terminal',instances=128,trials=384,cells=1536,regimes=xv,learning_pass=False,resident_gate=bool(xr['gates']['resident']['cache_random32']),cold_gate=bool(xr['gates']['cold']['cache_random32']),scope='E59 cold taskset removal; cold should route random32, resident can route cache_random32')
        paired = paired_launch_progress(repo)
        if paired is not None:
            result['paired_launch_followup'] = paired
        affinity_v2 = cache_affinity_v2_progress(repo)
        if affinity_v2 is not None:
            result['affinity_v2'] = affinity_v2
        ap=repo/'experiments/native_first_20260915_cache_affinity_probe.json'
        if ap.exists():
            ar=json.loads(ap.read_text())
            if ar.get('cache_hit_cells')!=768 or ar.get('cache_random32',{}).get('cells')!=1152 or ar.get('random32',{}).get('cells')!=1152:raise ValueError('cache affinity probe')
            result['affinity_probe']=dict(sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),cells=1152,cache_hit_cells=768,random32=dict(parse_mean_ms=n(ar['random32']['parse_mean_ms']),native_mean_ms=n(ar['random32']['native_mean_ms'])),cache_random32=dict(parse_mean_ms=n(ar['cache_random32']['parse_mean_ms']),native_mean_ms=n(ar['cache_random32']['native_mean_ms'])),scope='E55 affinity-corrected cache diagnostic; not full-cost confirmation')
        return result
    except (OSError,ValueError,KeyError,TypeError):return None

def load_performance_plan():
    raw=(HERE/'performance_plan.json').read_bytes()
    plan=json.loads(raw)
    if {g['id'] for g in plan['gaps']}!={f'P{i}' for i in range(1,8)} or len(plan['gaps'])!=7:
        raise ValueError('expected seven unique performance gaps')
    actions={a['id'] for a in plan['actions']}
    if len(actions)!=len(plan['actions']):
        raise ValueError('duplicate action ID')
    for gap in plan['gaps']:
        if gap['state'] not in ['部分改善','未解决','已解决'] or not set(gap['directions'])<=actions:
            raise ValueError('invalid gap state or action reference')
        for key in ['title','progress','remaining','source','evidence_level']:
            if not isinstance(gap[key],str) or not gap[key].strip():
                raise ValueError('missing gap description')
    seen=set()
    for action in plan['actions']:
        if not set(action['depends_on'])<=seen:
            raise ValueError('actions must be acyclic and dependency ordered')
        seen.add(action['id'])
        for key in ['title','priority','status','cost','hypothesis','method','gate','stop']:
            if not isinstance(action[key],str) or not action[key].strip():
                raise ValueError('missing action description')
    plan['sha256']=hashlib.sha256(raw).hexdigest()
    return plan

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

def overnight_progress(repo, now=None):
    """E61 duration/throughput only; never export job arguments or performance claims."""
    root=repo/'experiments/overnight_20260915_e61'
    try:
        frozen=(root/'CAMPAIGN.json').read_bytes();cfg=json.loads(frozen)
        state=json.loads((root/'STATUS.json').read_text())
        digest=hashlib.sha256(frozen).hexdigest()
        if state['frozen_sha256']!=digest or cfg['target_active_s']!=28800:
            return None
        def number(v):
            if type(v) not in (int,float) or not math.isfinite(v) or v<0:
                raise ValueError('invalid duration/count')
            return v
        def date(v):
            d=dt.datetime.fromisoformat(v)
            if d.tzinfo is None:raise ValueError('timestamp must include timezone')
            return d
        started=date(state['started_at']);observed=date(state['observed'])
        now=now or dt.datetime.now(dt.timezone.utc)
        age=(now-observed).total_seconds()
        active=number(state['active_s'])
        if active>(observed-started).total_seconds()+2:return None
        phases={'starting','running','paused_resources','draining','complete','stopped_incomplete'}
        if state['phase'] not in phases:return None
        if state['phase']=='complete' and active<cfg['target_active_s']:return None
        counts=dict(cpu_jobs=0,training_jobs=0,cells=0,training_steps=0)
        seen=set()
        for row in state['completed']:
            if row['id'] in seen:raise ValueError('duplicate completed job')
            seen.add(row['id'])
            folder=(root/row['job_path']).resolve().parent
            if not folder.is_relative_to(root.resolve()):raise ValueError('out of scope receipt')
            body=(folder/'RESULTS.json').read_bytes()
            if hashlib.sha256(body).hexdigest()!=row['result_sha256']:return None
            result=json.loads(body)
            if result['phase']!='terminal' or row['kind']!=result['kind']:return None
            if row['kind']=='cpu':
                counts['cpu_jobs']+=1;counts['cells']+=number(result['cells'])
            if row['kind']=='train':
                counts['training_jobs']+=1;counts['training_steps']+=number(result['steps'])
        running={kind:sum(r['kind']==kind for r in state['running']) for kind in ['cpu','train','data']}
        phase=state['phase']
        fresh=-5<=age<=120
        if not fresh and phase not in ['complete','stopped_incomplete']:phase='stale'
        return dict(phase=phase,observed=observed.isoformat(),started_at=started.isoformat(),
            active_s=active,target_active_s=cfg['target_active_s'],counts=counts,
            running=running,failed_jobs=len(state['failed']),training_queue=len(cfg['training_jobs']),
            cpu_slots=cfg['cpu_workers'],gpu_slots=len(cfg['gpus']),frozen_sha256=digest,
            duration_target_met=state['phase']=='complete' and active>=cfg['target_active_s'],
            heartbeat_fresh_at_build=fresh,
            earliest_target_at=(now+dt.timedelta(seconds=max(0,cfg['target_active_s']-active))).isoformat(),
            scope='探索实验吞吐与运行时长；不表示学习策略或端到端优势已成立')
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return None


def targeted_progress(repo):
    """Publish only terminal E62 engineering/readout aggregates with hash bindings."""
    root=repo/'experiments/targeted_20260915_e62'
    try:
        def read(name):return json.loads((root/name).read_text())
        def digest(name):return hashlib.sha256((root/name).read_bytes()).hexdigest()
        audit=read('AUDIT.json');raw=read('RESULTS.json');head=read('HEADROOM.json');stages=read('STAGES.json')
        if audit['verdict']!='accounting_and_sampled_witness_replay_pass':return None
        for name,key in [('RESULTS.json','results_sha256'),('HEADROOM.json','headroom_sha256'),
                         ('STAGES.json','stages_sha256'),('FROZEN.json','frozen_sha256'),('RECEIPT.json','receipt_sha256')]:
            if digest(name)!=audit[key]:return None
        if raw['cells']!=27648 or audit['cells']!=27648 or raw['receipt_sha256']!=audit['receipt_sha256']:return None
        def number(v):
            if type(v) not in (float,int) or not math.isfinite(v):raise ValueError('invalid number')
            return v
        totals={};contrasts={}
        for binary in ['old','popcnt']:
            for regime in ['cold','resident']:
                for arm in ['stock','degree32','random32','pair42','pair43','pair44']:
                    key=f'{binary}/{regime}/{arm}';v=raw['totals'][key]
                    if v['count']!=1152:return None
                    if any(type(v[k]) is not int or not 0<=v[k]<=1152 for k in ['solved','candidates']):return None
                    totals[key]={k:number(v[k]) for k in ['count','solved','candidates','par2_ms','median_ms','stdev_ms']}
                    if any(totals[key][k]<0 for k in totals[key]):return None
        for regime in ['cold','resident']:
            expected=True
            for arm in ['pair42','pair43','pair44']:
                key=f'repair/{regime}/{arm}'
                contrasts[key]={k:number(raw['contrasts'][key][k]) for k in ['gain_ms','simultaneous95_lower_ms']}
                t=raw['totals'][f'popcnt/{regime}/{arm}']
                for control in ['stock','degree32','random32']:
                    other=raw['totals'][f'popcnt/{regime}/{control}']
                    contrast=raw['contrasts'][f'learning/{regime}/{arm}/{control}']
                    expected &= t['par2_ms']<=.95*other['par2_ms'] and t['solved']>=other['solved'] and number(contrast['simultaneous95_lower_ms'])>0
                    expected &= all(t['scales'][n]['par2_ms']<=other['scales'][n]['par2_ms'] and t['scales'][n]['solved']>=other['scales'][n]['solved'] for n in ['24','32','48','64'])
            if raw['learning_gates'][regime] is not bool(expected):return None
        if head['count']!=4096 or sum(s['validation_count'] for s in head['shards'])!=4096:return None
        if head['additional']!=sum(s['additional'] for s in head['shards']):return None
        if type(head['additional']) is not int or not 0<=head['additional']<=4096:return None
        feature={b:{a:number(stages[f'{b}/resident/{a}']['features_s']['mean_us']) for a in ['pair42','pair43','pair44']} for b in ['old','popcnt']}
        residual=None
        rr=repo/'experiments/residual_20260915_e63'
        if (rr/'RESULTS.json').exists():
            rb=(rr/'RESULTS.json').read_bytes();rv=json.loads(rb);rs=json.loads((rr/'STATUS.json').read_text())
            if rs['phase']!='terminal' or rs['results_sha256']!=hashlib.sha256(rb).hexdigest():return None
            if hashlib.sha256((rr/'FROZEN.json').read_bytes()).hexdigest()!=rv['frozen_sha256']:return None
            if hashlib.sha256((rr/'rows.json.gz').read_bytes()).hexdigest()!=rv['rows_sha256']:return None
            counts={k:rv['counts'][k] for k in ['SAT','UNSAT','UNKNOWN']}
            if any(type(v) is not int or v<0 for v in counts.values()):return None
            if sum(counts.values())!=rv['residuals'] or rv['total_validation']!=4096:return None
            residual=dict(counts=counts,residuals=rv['residuals'],total_validation=4096,sha256=hashlib.sha256(rb).hexdigest())
        return dict(cells=27648,graphs=128,totals=totals,repair_contrasts=contrasts,residual=residual,
            learning_gates={r:raw['learning_gates'][r] for r in ['cold','resident']},
            headroom=dict(count=4096,additional=head['additional']),feature_mean_us=feature,
            audit_sha256=digest('AUDIT.json'),scope='同源码硬件指令修复的开发评测；不是独立确认或原论文学习优势')
    except (OSError,ValueError,KeyError,TypeError,AttributeError):return None


def exact_search_progress(repo):
    """Terminal-only E64 aggregates, including denial of unsupported advantage claims."""
    root=repo/'experiments/exact_search_20260915_e64'
    try:
        def read(name):return json.loads((root/name).read_text())
        def digest(name):return hashlib.sha256((root/name).read_bytes()).hexdigest()
        audit=read('AUDIT.json');raw=read('RESULTS.json');state=read('STATUS.json')
        if state['phase']!='terminal' or audit['verdict']!='complete_accounting_and_witness_replay_pass':return None
        for name in ['FROZEN.json','RESULTS.json','RECEIPT.json','TEST_RECEIPT.json','raw.jsonl.gz','charged.jsonl.gz']:
            if digest(name)!=audit[name.replace('.','_')+'_sha256']:return None
        if raw['receipt_sha256']!=audit['RECEIPT_json_sha256'] or state['receipt_sha256']!=raw['receipt_sha256']:return None
        if raw['cells']!=18432 or audit['cells']!=18432 or raw['learned_advantage'] is not False:return None
        def number(value):
            if type(value) not in (float,int) or not math.isfinite(value):raise ValueError('invalid number')
            return value
        def summary(value,count=None):
            n=value['count'];solved=value['solved']
            if type(n) is not int or n<=0 or (count is not None and n!=count):raise ValueError('bad count')
            if type(solved) is not int or not 0<=solved<=n:raise ValueError('bad solved')
            result={k:number(value[k]) for k in ['count','solved','par2_ms','median_ms','stdev_ms']}
            if any(v<0 for v in result.values()):raise ValueError('negative statistic')
            return result
        totals={};contrasts={}
        arms=['stock','degree32','random32','pair42','pair43','pair44','witness32','exact']
        for regime in ['cold','resident']:
            for arm in arms:
                key=f'{regime}/{arm}';value=raw['totals'][key]
                totals[key]=summary(value,1152)
                totals[key]['truth']={t:summary(value['truth'][t]) for t in ['10','20']}
                if sum(t['count'] for t in totals[key]['truth'].values())!=1152:return None
                totals[key]['scales']={n:summary(value['scales'][n],288) for n in ['24','32','48','64']}
                if sum(t['solved'] for t in totals[key]['scales'].values())!=value['solved']:return None
                if sum(t['solved'] for t in totals[key]['truth'].values())!=value['solved']:return None
            for control in ['stock','degree32','random32','witness32']:
                key=f'{regime}/{control}';value=raw['contrasts'][key]
                expected_graphs=totals[f'{regime}/exact']['truth']['20']['count']//9 if control=='witness32' else 128
                if type(value['graphs']) is not int or value['graphs']!=expected_graphs:return None
                contrasts[key]={k:number(value[k]) for k in ['graphs','gain_ms','simultaneous95_lower_ms']}
            target=totals[f'{regime}/exact'];expected=True
            for control in ['stock','degree32','random32']:
                other=totals[f'{regime}/{control}']
                expected &= target['par2_ms']<=.9*other['par2_ms'] and target['solved']>=other['solved']
                expected &= contrasts[f'{regime}/{control}']['simultaneous95_lower_ms']>0
                expected &= all(target['scales'][n]['par2_ms']<=other['scales'][n]['par2_ms'] and target['scales'][n]['solved']>=other['scales'][n]['solved'] for n in target['scales'])
            if raw['engineering_gates'][regime] is not bool(expected):return None
            a=target['truth']['20'];b=totals[f'{regime}/witness32']['truth']['20']
            expected=a['par2_ms']<=.9*b['par2_ms'] and a['solved']>=b['solved'] and contrasts[f'{regime}/witness32']['simultaneous95_lower_ms']>0
            if raw['unsat_mechanism_gates'][regime] is not bool(expected):return None
        return dict(cells=18432,graphs=128,totals=totals,contrasts=contrasts,
                    engineering_gates={r:raw['engineering_gates'][r] for r in ['cold','resident']},
                    unsat_mechanism_gates={r:raw['unsat_mechanism_gates'][r] for r in ['cold','resident']},
                    learned_advantage=False,sha256=digest('AUDIT.json'),scope='受限图编码的非学习精确搜索开发结果；不是独立确认、通用SAT或学习优势')
    except (OSError,ValueError,KeyError,TypeError,AttributeError):return None


def confirmation_progress(repo):
    """E65: one audited export bundle, three separate panels, recomputed gates."""
    root=repo/'experiments/confirm_20260915_e65'
    try:
        def read(name):return json.loads((root/name).read_text())
        audit=read('AUDIT.json');raw=read('RESULTS.json');state=read('STATUS.json')
        if audit['verdict']!='full_replay_pass' or state['phase']!='terminal':return None
        h=hashlib.sha256()
        for name in ['FROZEN.json','POOL.json','RECEIPT.json','RESULTS.json','raw.jsonl.gz','charged.jsonl.gz']:
            body=(root/name).read_bytes();label=name.encode()
            h.update(len(label).to_bytes(8,'big'));h.update(label);h.update(len(body).to_bytes(8,'big'));h.update(body)
        if audit['export_identity']!={'kind':'bundle_digest','value':h.hexdigest(),'scope':'release'}:return None
        if raw['input_identity']!=audit['input_identity'] or state['input_identity']!=raw['input_identity']:return None
        if raw['cells']!=19968 or audit['cells']!=19968 or raw['learned_advantage'] is not False:return None
        def number(v):
            if type(v) not in (float,int) or not math.isfinite(v):raise ValueError('invalid statistic')
            return v
        def summary(v,count=None):
            n=v['count'];s=v['solved']
            if type(n) is not int or n<0 or (count is not None and n!=count):raise ValueError('bad denominator')
            if type(s) is not int or not 0<=s<=n:raise ValueError('bad solved count')
            out=dict(count=n,solved=s)
            for k in ['par2_ms','median_ms','stdev_ms']:
                out[k]=None if n==0 else number(v[k])
                if n and out[k]<0:raise ValueError('negative statistic')
                if not n and v[k] is not None:raise ValueError('nonempty zero-count statistic')
            return out
        totals={};contrasts={};arms=['stock','degree32','random32','witness32','exact','pair42','pair43','pair44']
        for panel in ['confirm','fallback','stress']:
            count=1152 if panel=='confirm' else 96
            for regime in ['cold','resident']:
                for arm in arms if panel=='confirm' else ['stock','degree32','witness32','exact']:
                    key=f'{panel}/{regime}/{arm}';v=raw['totals'][key];out=summary(v,count)
                    out['truth']={t:summary(v['truth'][t]) for t in ['0','10','20']}
                    ns=['48','64'] if panel=='stress' else ['24','32','48','64']
                    out['scales']={n:summary(v['scales'][n],count//len(ns)) for n in ns}
                    if sum(x['count'] for x in out['truth'].values())!=count:return None
                    if sum(x['solved'] for x in out['truth'].values())!=out['solved']:return None
                    if sum(x['solved'] for x in out['scales'].values())!=out['solved']:return None
                    for k in ['recognized','graph_unknown']:
                        if type(v[k]) is not int or not 0<=v[k]<=count:return None
                        out[k]=v[k]
                    allowed=['legacy','no_response','exact_sat','exact_unsat','witness_sat','fallback']
                    out['routes']={k:v['routes'].get(k,0) for k in allowed}
                    if any(type(x) is not int or x<0 for x in out['routes'].values()) or sum(out['routes'].values())!=count:return None
                    totals[key]=out
        for regime in ['cold','resident']:
            keys=[f'confirm/{regime}/{c}' for c in ['stock','degree32','random32','witness32']]+[f'unsat/{regime}',f'fallback/{regime}']
            for key in keys:
                v=raw['contrasts'][key]
                expected=32 if key.startswith('fallback/') else totals[f'confirm/{regime}/exact']['truth']['20']['count']//9 if key.startswith('unsat/') else 128
                if type(v['graphs']) is not int or v['graphs']!=expected:return None
                contrasts[key]={'graphs':expected}
                for k in ['mean_ms','lower_ms','upper_ms']:
                    contrasts[key][k]=number(v[k]) if expected else None
                if expected and contrasts[key]['lower_ms']>contrasts[key]['upper_ms']:return None
            a=totals[f'confirm/{regime}/exact'];expected=True
            for c in ['stock','degree32','random32','witness32']:
                key=f'confirm/{regime}/{c}';b=totals[key]
                expected &= a['par2_ms']<=.9*b['par2_ms'] and a['solved']>=b['solved'] and contrasts[key]['lower_ms']>0
                expected &= all(a['scales'][n]['par2_ms']<=b['scales'][n]['par2_ms'] and a['scales'][n]['solved']>=b['scales'][n]['solved'] for n in a['scales'])
            if raw['confirmation_gates'][regime] is not bool(expected):return None
            a=totals[f'confirm/{regime}/exact']['truth']['20'];b=totals[f'confirm/{regime}/witness32']['truth']['20']
            expected=bool(a['count'] and a['par2_ms']<=.9*b['par2_ms'] and a['solved']>=b['solved'] and contrasts[f'unsat/{regime}']['lower_ms']>0)
            if raw['unsat_mechanism_gates'][regime] is not expected:return None
            a=totals[f'fallback/{regime}/exact'];b=totals[f'fallback/{regime}/stock']
            expected=bool(a['recognized']==0 and a['solved']>=b['solved'] and contrasts[f'fallback/{regime}']['upper_ms']<=.05*b['par2_ms'])
            if raw['fallback_noninferiority_gates'][regime] is not expected:return None
        return dict(run_id='E65',cells=19968,totals=totals,contrasts=contrasts,learned_advantage=False,
                    confirmation_gates={r:raw['confirmation_gates'][r] for r in ['cold','resident']},
                    unsat_mechanism_gates={r:raw['unsat_mechanism_gates'][r] for r in ['cold','resident']},
                    fallback_noninferiority_gates={r:raw['fallback_noninferiority_gates'][r] for r in ['cold','resident']},
                    scope='同主机新数据固定候选确认；压力/回退组分开，非学习或外部独立复现')
    except (OSError,ValueError,KeyError,TypeError,AttributeError):return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=HERE.parent)
    parser.add_argument('--out', type=Path, default=HERE / 'site')
    args = parser.parse_args()
    data = collect(args.repo.resolve())
    data['comparison']=comparison_progress(args.repo.resolve())
    data['performance_plan']=load_performance_plan()
    data['pilot']=pilot_progress(args.repo.resolve())
    data['hybrid']=hybrid_progress(args.repo.resolve())
    data['utility']=utility_progress(args.repo.resolve())
    data['conservative']=conservative_progress(args.repo.resolve())
    data['cost_aware']=conservative_progress(args.repo.resolve(),cost_aware=True)
    data['shared_gpu']=shared_gpu_progress(args.repo.resolve())
    data['structured_decoder']=structured_progress(args.repo.resolve())
    data['structured_policy']=structured_progress(args.repo.resolve(),small=True)
    data['resident_policy']=resident_progress(args.repo.resolve())
    data['decoder_utility']=decoder_utility_progress(args.repo.resolve())
    data['stability']=stability_progress(args.repo.resolve())
    data['native_first']=native_first_progress(args.repo.resolve())
    data['overnight']=overnight_progress(args.repo.resolve())
    data['targeted']=targeted_progress(args.repo.resolve())
    data['exact_search']=exact_search_progress(args.repo.resolve())
    data['confirmation']=confirmation_progress(args.repo.resolve())
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
