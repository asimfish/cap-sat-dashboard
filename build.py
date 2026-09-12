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

def conservative_progress(repo):
    """E29 aggregate contract: search trials are not independent instances."""
    root=repo/'experiments/conservative_search_20260913_e29'
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
        if state['frozen_sha256']!=h or state['phase'] not in ['running','failed','terminal'] or state['target_cells']!=18432:raise ValueError('E29 train identity')
        train={'phase':state['phase'],'observed':state['observed'],'completed_cells':num(state['completed_cells'],0,18432,True),'target_cells':18432,
            'workers':num(state['workers'],1,32,True),'frozen_sha256':h,'readout':None}
        if state['phase']=='terminal':
            if state['completed_cells']!=18432:raise ValueError('partial training terminal')
            raw=(root/'TRAINED.json').read_bytes();trained=json.loads(raw)
            if trained['train_frozen_sha256']!=h or trained['training_cells']!=18432:raise ValueError('trained identity')
            train['readout']={'sha256':hashlib.sha256(raw).hexdigest(),'summaries':{a:{'fitness_flips':num(trained['selected'][a]['fitness'],0,400000),
                'solved':num(trained['selected'][a]['solved'],0,192,True),'trials':num(trained['selected'][a]['trials'],192,192,True)} for a in ['cheap','cap']}}
        result={'training':train,'stages':{},'selection':None}
        for stage,instances in [('dev',96),('test',192)]:
            folder=root/stage
            if not (folder/'STATUS.json').exists():continue
            raw=(folder/'FROZEN.json').read_bytes();cfg=json.loads(raw);h=hashlib.sha256(raw).hexdigest();s=json.loads((folder/'STATUS.json').read_text());arms=cfg['arms'];n=instances*3
            allowed=['stock','random','hand','learned_cheap','learned_cap']
            if (stage=='dev' and arms!=allowed) or (stage=='test' and arms not in [allowed,allowed[:-1]]):raise ValueError('E29 stage arms')
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
            result['stages'][stage]=x
        if (root/'SELECTION.json').exists():
            s=json.loads((root/'SELECTION.json').read_text())
            if s['selected'] not in [None,'learned_cheap','learned_cap'] or s['development_sha256']!=hashlib.sha256((root/'dev/RESULTS.json').read_bytes()).hexdigest():raise ValueError('E29 selection')
            result['selection']={'selected':s['selected'],'stopped':s['selected'] is None}
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
