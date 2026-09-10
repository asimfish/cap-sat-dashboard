import copy
import json
from pathlib import Path
import tempfile
import unittest
from build import read_run, interval

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

if __name__=='__main__':
    unittest.main()
