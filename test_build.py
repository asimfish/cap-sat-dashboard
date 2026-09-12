import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from build import read_run, interval, native_progress, verify_terminal, comparison_progress, comparison_audit, load_performance_plan, pilot_progress, PILOT_ARMS, hybrid_progress, HYBRID_ARMS, utility_progress

class CollectorTests(unittest.TestCase):
    def test_utility_allowlist_and_terminal_identity(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/utility_ranker_20260913_e28';folder=root/'dev';folder.mkdir(parents=True)
            (root/'DATA_FROZEN.json').write_text('{}')
            trained=dict(data_frozen_sha256=hashlib.sha256(b'{}').hexdigest(),train_groups=4608,validation_groups=1536,
                models={a:dict(validation_gain_vs_minbreak=.04,parameters=723,private_weights=[1]) for a in ['utility_cheap','utility_cap']})
            (root/'TRAINED.json').write_text(json.dumps(trained));arms=['stock','random_repair','utility_cheap','utility_cap']
            frozen=json.dumps(dict(arms=arms));h=hashlib.sha256(frozen.encode()).hexdigest();(folder/'FROZEN.json').write_text(frozen)
            state=dict(phase='running',observed='2026-09-13T00:00:00+00:00',target_rows=24,target_cells=96,completed_rows=1,completed_cells=4,frozen_sha256=h,prepared_sha256='prep')
            (folder/'STATUS.json').write_text(json.dumps(state));out=utility_progress(repo)
            self.assertEqual(out['stages']['dev']['completed_cells'],4);self.assertNotIn('private',json.dumps(out));self.assertNotIn('test',out['stages'])
            state.update(phase='terminal',completed_rows=24,completed_cells=96);(folder/'STATUS.json').write_text(json.dumps(state))
            r=dict(rows=24,cells=96,frozen_sha256=h,prepared_sha256='prep',confirmed=False,errors=['private error'],
                summaries={a:dict(solved=5,par2_s=8,sls_solved=4) for a in arms})
            (folder/'RESULTS.json').write_text(json.dumps(r));out=utility_progress(repo)
            self.assertEqual(out['stages']['dev']['readout']['errors'],1);self.assertNotIn('private',json.dumps(out))
            r['confirmed']=True;(folder/'RESULTS.json').write_text(json.dumps(r));self.assertIsNone(utility_progress(repo))
            r['confirmed']=False;r['summaries']['stock']['par2_s']=float('nan')
            (folder/'RESULTS.json').write_text(json.dumps(r));self.assertIsNone(utility_progress(repo))

    def test_performance_plan_coverage_and_no_false_running(self):
        plan=load_performance_plan()
        self.assertEqual(len(plan['gaps']),7)
        self.assertEqual(len(plan['actions']),7)
        self.assertEqual(sum(g['state']=='部分改善' for g in plan['gaps']),4)
        self.assertEqual(sum(g['state']=='未解决' for g in plan['gaps']),3)
        actions={a['id']:a for a in plan['actions']}
        self.assertIn('E26',actions['A3']['status'])
        self.assertIn('E27',actions['A6']['status'])
        self.assertIn('E28',actions['A4']['status'])
        self.assertTrue(all(actions[a]['status']=='计划中 · 未启动' for a in ['A1','A2','A5']))
        self.assertRegex(plan['sha256'],r'^[0-9a-f]{64}$')
        payload=json.dumps(plan,ensure_ascii=False)
        for private in ['Bearer ','.whalent_tmp','/home/','ct-','Reviewer','Confidential']:
            self.assertNotIn(private,payload)

    def test_plan_rejects_missing_gap_and_unknown_action(self):
        plan=load_performance_plan()
        for mutated in [dict(plan,gaps=plan['gaps'][:-1]),copy.deepcopy(plan)]:
            if len(mutated['gaps'])==7:
                mutated['gaps'][0]['directions']=['unknown']
            with patch.object(Path,'read_bytes',return_value=json.dumps(mutated).encode()):
                with self.assertRaises(ValueError):load_performance_plan()

    def test_pilot_allowlist_and_terminal_guard(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/performance_pilot_20260912_e26';root.mkdir(parents=True)
            frozen=json.dumps({'arms':PILOT_ARMS,'cutoff_wall_s':10,'private':'/private/checkpoint'})
            (root/'FROZEN.json').write_text(frozen)
            status=dict(phase='running',started='2026-09-12T08:00:00+00:00',target_rows=24,target_cells=144,completed_rows=1,completed_cells=6,resources={'host':'private'},pid=123)
            def save_status(): (root/'STATUS.json').write_text(json.dumps(status))
            save_status();result=pilot_progress(repo)
            self.assertEqual(result['completed_cells'],6)
            self.assertNotIn('private',json.dumps(result));self.assertNotIn('pid',json.dumps(result))
            status['phase']='terminal';save_status();self.assertIsNone(pilot_progress(repo))
            status.update(completed_rows=24,completed_cells=144);save_status()
            self.assertEqual(pilot_progress(repo)['phase'],'awaiting_readout')
            report=dict(frozen_sha256=hashlib.sha256(frozen.encode()).hexdigest(),rows=24,cells=144,status='no_confirmed_positive_effect',errors=['private error'],walksat_budget_failures=[],unverified_unsat_cells=0,
                summaries={a:dict(verified_par2_s=12,reported_par2_s=11,verified_solved=10,entered=20,released=18) for a in PILOT_ARMS},
                contrasts={a:dict(mean_gain_s=-2,simultaneous95_band=[-5,1],gate=False) for a in ['stock','segmented','pol_pulse','ws_pulse']})
            (root/'RESULTS.json').write_text(json.dumps(report));result=pilot_progress(repo)
            self.assertEqual(result['readout']['errors'],1);self.assertNotIn('private',json.dumps(result))
            report['status']='promising_requires_independent_confirmation'
            (root/'RESULTS.json').write_text(json.dumps(report));self.assertIsNone(pilot_progress(repo))
            report['status']='no_confirmed_positive_effect'
            report['summaries']['stock']['verified_par2_s']=float('nan')
            (root/'RESULTS.json').write_text(json.dumps(report));self.assertIsNone(pilot_progress(repo))

    def test_hybrid_separates_stages_and_omits_private_fields(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/performance_hybrid_20260912_e27';folder=root/'dev';folder.mkdir(parents=True)
            frozen=json.dumps({'arms':HYBRID_ARMS,'private_path':'/private/model'});h=hashlib.sha256(frozen.encode()).hexdigest()
            (folder/'FROZEN.json').write_text(frozen)
            status=dict(phase='running',observed='2026-09-12T10:00:00+00:00',target_rows=24,target_cells=120,
                completed_rows=2,completed_cells=10,frozen_sha256=h,prepared_sha256='prepared',resources={'private':'host'})
            def save_status(): (folder/'STATUS.json').write_text(json.dumps(status))
            save_status();r=hybrid_progress(repo)
            self.assertEqual(r['stages']['dev']['completed_cells'],10);self.assertNotIn('test',r['stages']);self.assertNotIn('private',json.dumps(r))
            status['phase']='terminal';save_status();self.assertIsNone(hybrid_progress(repo))
            status.update(completed_rows=24,completed_cells=120);save_status();self.assertEqual(hybrid_progress(repo)['stages']['dev']['phase'],'awaiting_readout')
            def summaries(n):return {a:dict(verified_par2_s=12,reported_par2_s=12,verified_solved=n//2,sls_solved=n//4,fallback_solved=n//4) for a in HYBRID_ARMS}
            report=dict(frozen_sha256=h,prepared_sha256='prepared',rows=24,cells=120,status='development_complete',engineering_pass=False,learning_pass=False,
                        summaries=summaries(24),by_scale={s:summaries(12) for s in ['325','500']},errors=['private failure'],unverified_unsat_cells=0,contrasts={})
            def save_report(): (folder/'RESULTS.json').write_text(json.dumps(report))
            save_report();r=hybrid_progress(repo)
            self.assertEqual(r['stages']['dev']['readout']['errors'],1);self.assertNotIn('private',json.dumps(r))
            report['engineering_pass']=True;save_report();self.assertIsNone(hybrid_progress(repo))
            report['engineering_pass']=False;report['summaries']['stock']['sls_solved']=999;save_report();self.assertIsNone(hybrid_progress(repo))
            report['summaries']['stock']['sls_solved']=6
            report['summaries']['stock']['verified_par2_s']=float('nan')
            save_report();self.assertIsNone(hybrid_progress(repo))

    def test_plan_rejects_cycles_and_missing_gate(self):
        plan=load_performance_plan()
        for mode in ['cycle','gate']:
            mutated=copy.deepcopy(plan)
            if mode=='cycle':mutated['actions'][0]['depends_on']=['A5']
            else:mutated['actions'][0]['gate']=''
            with patch.object(Path,'read_bytes',return_value=json.dumps(mutated).encode()):
                with self.assertRaises(ValueError):load_performance_plan()

    def test_hybrid_confirmation_verdict_and_selection_identity(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/performance_hybrid_20260912_e27';folder=root/'test';folder.mkdir(parents=True)
            selected=json.dumps(dict(cap='cap_warm_repair',engineering='random_repair'));(root/'SELECTION.json').write_text(selected)
            arms=HYBRID_ARMS[:4];frozen=json.dumps(dict(arms=arms,selection_sha256=hashlib.sha256(selected.encode()).hexdigest()))
            (folder/'FROZEN.json').write_text(frozen);h=hashlib.sha256(frozen.encode()).hexdigest()
            status=dict(phase='terminal',observed='2026-09-12T10:00:00+00:00',target_rows=48,target_cells=192,
                completed_rows=48,completed_cells=192,frozen_sha256=h,prepared_sha256='prepared')
            (folder/'STATUS.json').write_text(json.dumps(status))
            def summaries(n):return {a:dict(verified_par2_s=10,reported_par2_s=10,verified_solved=n//2,sls_solved=n//4,fallback_solved=n//4) for a in arms}
            c=dict(arm='random_repair',control='stock',mean_gain_s=2,simultaneous95_band=[.1,4],gate=True)
            report=dict(frozen_sha256=h,prepared_sha256='prepared',rows=48,cells=192,status='local_engineering_improvement_only',engineering_pass=True,learning_pass=False,
                        summaries=summaries(48),by_scale={s:summaries(24) for s in ['325','500']},errors=[],unverified_unsat_cells=0,contrasts={'engineering_vs_stock':c})
            def save(): (folder/'RESULTS.json').write_text(json.dumps(report))
            save();r=hybrid_progress(repo);self.assertFalse(r['stages']['test']['readout']['engineering_pass']);self.assertEqual(r['stages']['test']['readout']['qualification'],'awaiting_matched_guard')
            matched=root/'matched';matched.mkdir();(matched/'FROZEN.json').write_text('{}')
            joint=dict(main_results_sha256=hashlib.sha256((folder/'RESULTS.json').read_bytes()).hexdigest(),matched_frozen_sha256=hashlib.sha256(b'{}').hexdigest(),
                       matched_executed=False,engineering_pass=True,learning_pass=False,errors=[],contrasts={'engineering_vs_stock':c})
            (root/'JOINT_RESULTS.json').write_text(json.dumps(joint));r=hybrid_progress(repo)
            self.assertTrue(r['stages']['test']['readout']['engineering_pass']);self.assertFalse(r['stages']['test']['readout']['learning_pass'])
            report['status']='local_learning_improvement';save();self.assertIsNone(hybrid_progress(repo))
            report['status']='local_engineering_improvement_only';save();(root/'SELECTION.json').write_text(selected+' ')
            self.assertIsNone(hybrid_progress(repo))

    def test_terminal_aggregate_allowlist(self):
        cell={'n':300,'verified_solved':290,'unverified_unsat':10,'reported_par2_s':2,'verified_par2_s':3,'private_path':'/secret'}
        backends={b:{a:cell for a in ['stock','native','capsat_standing_adapter','polarity_initial','polarity_standing','walksat_standing']} for b in ['glucose','kissat']}
        raw={'complete_denominator':True,'completed_rows':619,'recorded_cells':7428,'observed_unix':1789091299,'issues':[],'failures':[],'unverified_unsat_cells':['private'], 'groups':{c:{'ALL':{'n':300,'backends':backends},'UNKNOWN':{'n':0,'backends':{}}} for c in ['fresh_n200','existing_n350','existing_industrial19']}}
        with patch.object(Path,'read_bytes',return_value=json.dumps(raw).encode()):
            result=comparison_audit(Path('/repo'))
        self.assertEqual(len(result['rows']),36)
        self.assertNotIn('private',json.dumps(result))
        cell['verified_par2_s']=float('nan')
        with patch.object(Path,'read_bytes',return_value=json.dumps(raw).encode()):
            self.assertIsNone(comparison_audit(Path('/repo')))

    def read(self, records):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'solve_cadical').mkdir()
            (root/'solve_cadical/n350_seed42_cadical195_shard0of10.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            return read_run(root,42)

    def row(self):
        return {'record_type':'solve_row','row_index':0,'arms':{a:{'satisfiable':True,'unsatisfiable':False,'full_pipeline_par2_s':10.0} for a in ['capsat','default','polarity','walksat_equal']}}

    def test_invalid_measurements_excluded(self):
        for invalid in [float('nan'),float('inf'),-1,True,'10']:
            r=self.row()
            r['arms']['capsat']['full_pipeline_par2_s']=invalid
            rows,errors,_=self.read([r])
            self.assertEqual((len(rows),errors),(0,1))

    def test_duplicate_and_conflicting_rows(self):
        r=self.row()
        self.assertEqual(self.read([r,r])[:2],({0:r},0))
        other=copy.deepcopy(r)
        other['arms']['capsat']['full_pipeline_par2_s']=11
        self.assertEqual(self.read([r,other])[1],1)

    def test_bad_record_and_solver_error(self):
        r=self.row()
        r['arm_errors']={'capsat':'failed'}
        rows,errors,_=self.read([[],r])
        self.assertEqual((len(rows),errors),(0,2))

    def test_interval_sign_and_repeatability(self):
        result=interval([-3,-2,-1])
        self.assertEqual(result,interval([-3,-2,-1]))
        self.assertEqual(result[0],-2)
        self.assertLess(result[2],0)

    def test_native_export_omits_private_inventory(self):
        raw={'time_utc':'2026-09-10T00:00:00Z','stage':'rlaf_verified_v2_running',
             'pid':123,'command':'private command','owned_children':[{'pid':456}],
             'progress':{'completed_iterations':49,'target_iterations':100,'path':'/private'},
             'preparation':{'label_records':200,'target_records':300,'pid':456},
             'alerts':['VALID_ALERT','/private/path']}
        with patch.object(Path,'read_text',return_value=json.dumps(raw)):
            result=native_progress(Path('/repo'))
        payload=json.dumps(result)
        self.assertNotIn('private',payload)
        self.assertNotIn('pid',payload)
        self.assertEqual(result['progress']['completed_iterations'],49)
        self.assertEqual(result['preparation']['label_records'],200)
        self.assertEqual(result['alerts'],['VALID_ALERT'])

    def test_terminal_hash_verification_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            files={'rlaf_verified_v2.log':'Optimized model for 50 steps\n'*100,
                   'rlaf_verified_v2/best.pt':'fixture',
                   'neuroback_small/training.jsonl':'\n'.join(json.dumps({'epoch':i}) for i in range(40)),
                   'neuroback_small/COMPLETE.json':'{}','neuroback_small/best.ptg':'fixture'}
            for name,content in files.items():
                p=root/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(content)
            raw={'terminal':True,'acceptance':{}}
            for name,prefix in [('rlaf_verified_v2','rlaf'),('neuroback_verified_v2','neuroback')]:
                (root/(name+'_RECEIPT.json')).write_text('{"returncode":0}')
                raw['acceptance'][name]={'status':'training_artifacts_verified','hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items() if k.startswith(prefix)}}
            status,counts=verify_terminal(root,raw)
            self.assertEqual(status,'verified');self.assertEqual(counts['completed_epochs'],40)
            (root/'rlaf_verified_v2/best.pt').write_text('changed')
            self.assertEqual(verify_terminal(root,raw)[0],'not_verified')

    def test_comparison_export_omits_runtime_identity(self):
        raw={'time_unix':1789071033,'state':{'phase':'solving','controller_pid':12,'controller_command':['private']},
             'completed_paired_rows':300,'target_paired_rows':619,'completed_arm_cells':3600,'target_arm_cells':7428,
             'alerts':['PIPELINE_ERROR','/private/path']}
        with patch.object(Path,'read_text',return_value=json.dumps(raw)),patch.object(Path,'exists',return_value=False):
            result=comparison_progress(Path('/repo'))
        self.assertEqual(result['completed_arm_cells'],3600)
        self.assertEqual(result['phase'],'solving')
        self.assertEqual(result['alerts'],['PIPELINE_ERROR'])
        self.assertNotIn('private',json.dumps(result))
        self.assertNotIn('pid',json.dumps(result))

if __name__=='__main__':
    unittest.main()
