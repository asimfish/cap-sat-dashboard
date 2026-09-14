import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build import paired_launch_progress


class PairedLaunchTests(unittest.TestCase):
    def test_audit_binding_exact_arms_finite_costs_and_no_private_fields(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            folder = repo / 'experiments/native_first_20260915_e60/run'
            folder.mkdir(parents=True)
            frozen = folder / 'FROZEN.json'
            frozen.write_text('{"private":"/home/private"}')
            fh = hashlib.sha256(frozen.read_bytes()).hexdigest()
            result = dict(cells=3456, graphs=128, trials_per_arm=1152, output_mismatches=0,
                          repeats=3, frozen_sha256=fh, independent_confirmation=False,
                          learned_policy_advantage=False,
                          summaries={a: dict(count=1152, solved=1152, errors=0, par2_ms=cost,
                              raw_s={'median':cost}, setup_s={'mean':cost/2},
                              native_wall_ms={'mean':2}, audited_total_s={'mean':cost+1})
                              for a, cost in [('taskset',18), ('inherited',13), ('unbound',11)]},
                          contrasts={'inherited_vs_taskset': dict(gain_ms=5, gain_pct=100*5/18,
                              simultaneous95_lower_ms=4, simultaneous95_upper_ms=6)},
                          by_scale={str(n): {a: {'par2_ms':1} for a in ['taskset','inherited','unbound']}
                                    for n in [24,32,48,64]}, private='/home/private', pid=123)
            def save(x):
                p = folder / 'RESULTS.json'
                p.write_text(json.dumps(x))
                (folder / 'AUDIT.json').write_text(json.dumps(dict(verdict='integrity_pass',
                    cells=3456, checked_masks=3456, output_mismatches=0,
                    results_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), frozen_sha256=fh)))
            self.assertIsNone(paired_launch_progress(repo))
            save(result)
            out = paired_launch_progress(repo)
            self.assertEqual(out['regimes']['inherited']['par2_ms'], 13)
            self.assertEqual(out['comparison']['gain_ms'], 5)
            self.assertNotIn('/home/', json.dumps(out))
            self.assertNotIn('pid', json.dumps(out))
            (folder / 'RESULTS.json').write_text('{}')
            self.assertIsNone(paired_launch_progress(repo))
            for change in ['nan', 'fake_arm', 'mismatch', 'confirmation']:
                bad = copy.deepcopy(result)
                if change == 'nan': bad['summaries']['inherited']['par2_ms'] = float('nan')
                if change == 'fake_arm': bad['summaries']['degree32'] = bad['summaries']['taskset']
                if change == 'mismatch': bad['output_mismatches'] = 1
                if change == 'confirmation': bad['independent_confirmation'] = True
                save(bad)
                self.assertIsNone(paired_launch_progress(repo), change)


if __name__ == '__main__':
    unittest.main()
