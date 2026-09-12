import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from build import read_run, interval, native_progress, verify_terminal, comparison_progress, comparison_audit, load_performance_plan, pilot_progress, PILOT_ARMS

class CollectorTests(unittest.TestCase):
    def test_performance_plan_coverage_and_no_false_running(self):
        plan=load_performance_plan()
        self.assertEqual(len(plan['gaps']),7)
        self.assertEqual(len(plan['actions']),6)
        self.assertEqual(sum(g['state']=='部分改善' for g in plan['gaps']),4)
        self.assertEqual(sum(g['state']=='未解决' for g in plan['gaps']),3)
        actions={a['id']:a for a in plan['actions']}
        self.assertIn('E26',actions['A3']['status'])
        self.assertTrue(all(actions[a]['status']=='计划中 · 未启动' for a in ['A1','A2','A4','A5']))
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

    def test_plan_rejects_cycles_and_missing_gate(self):
        plan=load_performance_plan()
        for mode in ['cycle','gate']:
            mutated=copy.deepcopy(plan)
            if mode=='cycle':mutated['actions'][0]['depends_on']=['A5']
            else:mutated['actions'][0]['gate']=''
            with patch.object(Path,'read_bytes',return_value=json.dumps(mutated).encode()):
                with self.assertRaises(ValueError):load_performance_plan()

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
