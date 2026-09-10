import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from build import read_run, interval, native_progress, verify_terminal, comparison_progress

class CollectorTests(unittest.TestCase):
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
