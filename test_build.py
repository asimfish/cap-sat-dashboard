import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from build import stability_progress, overnight_progress, targeted_progress, exact_search_progress, confirmation_progress, recognition_repair_progress, prefix_campaign_progress, joint_feedback_progress, local_feedback_progress
from unittest.mock import patch
from build import compact_distill_progress, boundary_screen_progress, coverage_screen_progress, night_recovery_progress, recovery_v3_progress, depth_screen_progress, literal_fullcost_progress, weight_precision_progress, head_refit_progress, strategy_headroom_progress
from build import read_run, interval, native_progress, verify_terminal, comparison_progress, comparison_audit, load_performance_plan, pilot_progress, PILOT_ARMS, hybrid_progress, HYBRID_ARMS, utility_progress, conservative_progress, shared_gpu_progress, structured_progress, resident_progress, decoder_utility_progress

class CollectorTests(unittest.TestCase):
    def test_strategy_recovery_requires_bound_reuse_audit(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/strategy_headroom_20260918_v2';root.mkdir(parents=True)
            def save(name,value):(root/name).write_text(json.dumps(value))
            base=dict(kind='bundle_digest',scope='strategy_headroom_inputs',value='a'*64)
            ident=dict(kind='bundle_digest',scope='strategy_headroom_recovery_inputs',value='b'*64)
            cfg=dict(rows=[{}]*192,cells=768,predictions=384,arms=['degree','teacher'],end_unix=1789673400,base_identity=base,reused=[{}]*97,remaining=[{}]*287)
            state=dict(input_identity=ident,base_identity=base,phase='complete',stage='audit',observed_unix=1000.,elapsed_s=20.,window_end_unix=1789673400,progress=dict(stage='solve_blocks',completed=384,target=384),reused_cells=194,new_target_cells=574,cleanup=dict(errors=[],unreaped=[]),receipts=[dict(label=s,returncode=0) for s in ('predict','reference','solve','audit')])
            save('CONFIG.json',cfg);save('FROZEN.json',dict(input_identity=ident));save('STATUS.json',state);save('FINAL.json',state)
            save('AUDIT.json',dict(input_identity=base,errors=[],cells=768,predictions=384))
            result=dict(input_identity=base,cells=768,end_to_end_advantage=False,holdout_opened=False,offline_selection_cost_excluded=True,oracle_space_pass=True,size_rule_pass=False,scales=dict(ALL={a:dict(cells=384,solved=100,par2_s=3.) for a in ('teacher','degree','oracle','crossfit','cross_repeat')}))
            save('RESULTS.json',result);self.assertIsNone(strategy_headroom_progress(repo,1000))
            proof=dict(input_identity=ident,base_identity=base,errors=[],reused_cells=194,new_cells=574,result_sha256=hashlib.sha256((root/'RESULTS.json').read_bytes()).hexdigest(),audit_sha256=hashlib.sha256((root/'AUDIT.json').read_bytes()).hexdigest())
            save('RECOVERY_AUDIT.json',proof);value=strategy_headroom_progress(repo,1000);self.assertTrue(value['audited']);self.assertEqual(value['reused_cells'],194)
            proof['new_cells']=575;save('RECOVERY_AUDIT.json',proof);self.assertIsNone(strategy_headroom_progress(repo,1000))

    def test_strategy_headroom_not_deployment_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/strategy_headroom_20260918_v1';root.mkdir(parents=True)
            def save(name,value):(root/name).write_text(json.dumps(value))
            ident=dict(kind='bundle_digest',scope='strategy_headroom_inputs',value='f'*64)
            cfg=dict(rows=[{}]*192,cells=768,predictions=384,arms=['degree','teacher'],end_unix=1789673400)
            state=dict(input_identity=ident,phase='running',stage='solve',observed_unix=1000.,elapsed_s=20.,window_end_unix=1789673400,progress=dict(stage='solve_blocks',completed=100,target=384),supervisor=dict(private='/home/PRIVATE'))
            save('CONFIG.json',cfg);save('FROZEN.json',dict(input_identity=ident));save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True):
                result=strategy_headroom_progress(repo,1000);self.assertEqual(result['completed'],100);self.assertFalse(result['audited']);self.assertNotIn('PRIVATE',json.dumps(result))
                self.assertEqual(strategy_headroom_progress(repo,1121)['phase'],'stale')
                state.update(phase='complete',stage='audit',cleanup=dict(errors=[],unreaped=[]),receipts=[dict(label=s,returncode=0) for s in ('predict','reference','solve','audit')]);save('STATUS.json',state);save('FINAL.json',state)
                self.assertIsNone(strategy_headroom_progress(repo,9999))
                save('AUDIT.json',dict(input_identity=ident,errors=[],cells=768,predictions=384))
                result=dict(input_identity=ident,cells=768,end_to_end_advantage=False,holdout_opened=False,offline_selection_cost_excluded=True,oracle_space_pass=True,size_rule_pass=False,scales=dict(ALL={a:dict(cells=384,solved=100,par2_s=3.) for a in ('teacher','degree','oracle','crossfit','cross_repeat')}))
                save('RESULTS.json',result);out=strategy_headroom_progress(repo,9999);self.assertTrue(out['audited']);self.assertFalse(out['learned_advantage'])
                result['scales']['ALL']['teacher']['par2_s']=float('nan');save('RESULTS.json',result);self.assertIsNone(strategy_headroom_progress(repo,9999))

    def test_head_refit_receipts_stale_counts_and_no_promotion(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/head_refit_20260918_v1';root.mkdir(parents=True)
            def save(name,value):(root/name).write_text(json.dumps(value))
            labels=[f'{a}_{s}' for a in ('original','balanced') for s in (42,43,44)];ident=dict(kind='bundle_digest',scope='head_refit_inputs',value='e'*64)
            cfg=dict(lanes=[dict(label=l) for l in labels],target_fits=6,records=21024,end_unix=1789673400)
            state=dict(input_identity=ident,phase='running',stage='fitting',observed_unix=1000.,elapsed_s=20.,window_end_unix=1789673400,lanes=[],completed_fits=0,target_fits=6,supervisor=dict(private='/home/PRIVATE'))
            save('CONFIG.json',cfg);save('FROZEN.json',dict(input_identity=ident));save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True):
                result=head_refit_progress(repo,1000);self.assertEqual(result['completed_fits'],0);self.assertFalse(result['audited']);self.assertNotIn('PRIVATE',json.dumps(result))
                self.assertEqual(head_refit_progress(repo,1121)['phase'],'stale')
                state.update(phase='complete',stage='summary',completed_fits=6,lanes=[dict(label=l,captured=1752,target_formulas=1752,completed_fits=1,phase='complete') for l in labels],cleanup=dict(errors=[],unreaped=[]),receipts=[dict(label=f'{s}_{l}',returncode=0) for s in ('fit','audit') for l in labels]+[dict(label='summary',returncode=0)])
                save('STATUS.json',state);save('FINAL.json',state);self.assertFalse(head_refit_progress(repo,9999)['audited'])
                save('AUDIT.json',dict(input_identity=ident,errors=[],fits=6,records=21024))
                save('RESULTS.json',dict(input_identity=ident,records=21024,solver_cells=0,end_to_end_advantage=False,independently_confirmed=False,all_seeds_pass=False))
                result=head_refit_progress(repo,9999);self.assertTrue(result['audited']);self.assertFalse(result['repair_screen']);self.assertFalse(result['learned_advantage'])
                state['receipts'].pop();save('STATUS.json',state);save('FINAL.json',state);self.assertIsNone(head_refit_progress(repo,9999))

    def test_weight_precision_no_premature_promotion_or_private_fields(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/weight_precision_20260918_v1';root.mkdir(parents=True)
            def save(name,value):(root/name).write_text(json.dumps(value))
            labels=[f'{a}_{s}' for a in ('original','balanced') for s in (42,43,44)];ident=dict(kind='bundle_digest',scope='weight_precision_inputs',value='d'*64)
            cfg=dict(lanes=[dict(label=l) for l in labels],steps=3072,total_updates=18432,end_unix=1789665600)
            state=dict(input_identity=ident,phase='running',observed_unix=1000.,elapsed_s=20.,window_end_unix=1789665600,training=[],completed_updates=0,target_updates=18432,supervisor=dict(private='/home/PRIVATE'))
            save('CONFIG.json',cfg);save('FROZEN.json',dict(input_identity=ident));save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True):
                out=weight_precision_progress(repo,1000);self.assertEqual(out['target_updates'],18432);self.assertFalse(out['audited']);self.assertNotIn('PRIVATE',json.dumps(out))
                self.assertEqual(weight_precision_progress(repo,1121)['phase'],'stale')
                state.update(phase='complete',completed_updates=18432,training=[dict(label=l,updates=3072,target_updates=3072,phase='complete') for l in labels],receipts=[dict(label=l,returncode=0) for l in labels],cleanup=dict(errors=[],unreaped=[]));save('STATUS.json',state);save('FINAL.json',state)
                self.assertFalse(weight_precision_progress(repo,9999)['audited'])
                save('AUDIT.json',dict(input_identity=ident,errors=[],updates=18432,records=14256));self.assertTrue(weight_precision_progress(repo,9999)['audited'])
                state['completed_updates']=18431;save('STATUS.json',state);self.assertIsNone(weight_precision_progress(repo,9999))

    def test_literal_fullcost_counts_terminal_stale_and_privacy(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/literal_fullcost_20260917_v1';root.mkdir(parents=True)
            def save(name,value):(root/name).write_text(json.dumps(value))
            arms=['stock','polarity','degree','teacher','d6_raw_42','d6_raw_43','d6_raw_44'];ident=dict(kind='bundle_digest',scope='literal_fullcost_inputs',value='c'*64)
            cfg=dict(arms=arms,rows=[{}]*144,prediction_requests=1152,target_cells=2016,end_unix=1789663200)
            state=dict(input_identity=ident,phase='running',stage='evaluation',observed_unix=1000.,elapsed_s=20.,completed_cells=7,target_cells=2016,completed_predictions=1152,target_predictions=1152,receipts=[],supervisor=dict(private='/home/PRIVATE'))
            save('CONFIG.json',cfg);save('FROZEN.json',dict(input_identity=ident));save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True):
                out=literal_fullcost_progress(repo,1000);self.assertEqual(out['completed_cells'],7);self.assertFalse(out['audited']);self.assertNotIn('PRIVATE',json.dumps(out))
                self.assertEqual(literal_fullcost_progress(repo,1121)['phase'],'stale')
                state['completed_cells']=8;save('STATUS.json',state);self.assertIsNone(literal_fullcost_progress(repo,1000))
                state.update(phase='complete',completed_cells=2016,receipts=[dict(arm=a,returncode=0) for a in arms[3:]]);save('STATUS.json',state);save('FINAL.json',dict(state,cleanup=dict(errors=[],unreaped=[])))
                self.assertFalse(literal_fullcost_progress(repo,9999)['audited'])
                raw=dict(input_identity=ident,cells=2016,end_to_end_advantage=False,independently_confirmed=False,holdout_opened=False,family_feasible=False,family_potential_advantage=False,scales={s:{a:dict(cells=288 if s=='ALL' else 96,solved=50,mean_par2_s=3.) for a in arms} for s in ('ALL','200','300','350')})
                save('RESULTS.json',raw);save('AUDIT.json',dict(input_identity=ident,cells=2016,predictions=1152,errors=[]));self.assertTrue(literal_fullcost_progress(repo,9999)['audited'])
                raw['scales']['350']['teacher']['mean_par2_s']=float('nan');save('RESULTS.json',raw);self.assertIsNone(literal_fullcost_progress(repo,9999))

    def test_depth_progress_rejects_counts_and_does_not_promote_partial(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/depth_literal_20260917_v1';root.mkdir(parents=True)
            def save(name,value):(root/name).write_text(json.dumps(value))
            labels=[f'{p}_{s}' for p in ('d6_mixed','d6_raw','d10_mixed','d10_raw') for s in (42,43,44)]
            ident=dict(kind='bundle_digest',scope='literal_depth_inputs',value='b'*64)
            cfg=dict(lanes=[dict(label=l) for l in labels],steps=9216,total_updates=110592,end_unix=1789663200)
            state=dict(input_identity=ident,phase='running',observed_unix=1000.,elapsed_s=20.,window_end_unix=1789663200,training=[dict(label=labels[0],updates=64,phase='training',target_updates=9216,private='/home/PRIVATE')],completed_updates=64,target_updates=110592,supervisor=dict(private='PRIVATE'))
            save('CONFIG.json',cfg);save('FROZEN.json',dict(input_identity=ident));save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True):
                result=depth_screen_progress(repo,1000);self.assertEqual(result['completed_updates'],64);self.assertFalse(result['audited']);self.assertFalse(result['learned_advantage']);self.assertNotIn('PRIVATE',json.dumps(result))
                self.assertEqual(depth_screen_progress(repo,1121)['phase'],'stale')
                state['completed_updates']=128;save('STATUS.json',state);self.assertIsNone(depth_screen_progress(repo,1000))
                state.update(phase='complete',completed_updates=110592,training=[dict(label=l,updates=9216,phase='complete',target_updates=9216) for l in labels],cleanup=dict(errors=[],unreaped=[]),receipts=[dict(label=l,returncode=0) for l in labels]);save('STATUS.json',state);save('FINAL.json',state)
                self.assertFalse(depth_screen_progress(repo,9999)['audited'])
                save('AUDIT.json',dict(input_identity=ident,errors=[],updates=110592,records=28512))
                self.assertTrue(depth_screen_progress(repo,9999)['audited'])
    def test_recovery_v3_counts_stale_terminal_and_privacy(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/anneal_literal_20260917_v3';root.mkdir(parents=True)
            def save(name,value):(root/name).write_text(json.dumps(value))
            labels=[f'{p}_{arm}_{seed}' for p in ('lit_mixed','lit_raw') for arm in ('constant','anneal') for seed in (42,43,44)]
            ident=dict(kind='bundle_digest',scope='literal_anneal_recovery_v3_inputs',value='a'*64)
            lanes=[dict(label=l,origin='reused' if i<6 else 'new') for i,l in enumerate(labels)]
            cfg=dict(end_unix=1789663200,lanes=lanes);state=dict(input_identity=ident,window_end_unix=1789663200,phase='running_owner',current='owner.py',observed_unix=1000.,error=None,supervisor=dict(private='/home/PRIVATE'),receipts=[])
            training=[dict(label=l,updates=3072,target_updates=3072,phase='complete') for l in labels[:6]]
            owner=dict(input_identity=ident,window_end_unix=1789663200,phase='running',new_updates=0,reused_updates=18432,completed_updates=18432,target_updates=36864,training=training)
            save('FROZEN.json',dict(input_identity=ident));save('CONFIG.json',cfg);save('SUPERVISION_STATUS.json',state);save('STATUS.json',owner)
            with patch('build.joint_supervisor_alive',return_value=True):
                result=recovery_v3_progress(repo,1000);self.assertEqual(result['recovery']['new_updates'],0);self.assertTrue(result['prior_window_ended']);self.assertNotIn('PRIVATE',json.dumps(result));self.assertFalse(result['learned_advantage'])
                self.assertEqual(recovery_v3_progress(repo,1121)['phase'],'stale')
                owner['new_updates']=64;save('STATUS.json',owner);self.assertIsNone(recovery_v3_progress(repo,1000))
                owner.update(new_updates=18432,completed_updates=36864,phase='complete',cleanup=dict(errors=[],unreaped=[]),receipts=[dict(label=l,returncode=0) for l in labels],training=[dict(label=l,updates=3072,target_updates=3072,phase='complete') for l in labels]);save('STATUS.json',owner);save('FINAL.json',owner)
                state.update(phase='complete',current=None,receipts=[dict(script=s,returncode=0) for s in ('owner.py','audit.py')]);save('SUPERVISION_STATUS.json',state);save('SUPERVISION_FINAL.json',dict(state,cleanup=dict(errors=[],unreaped=[])))
                self.assertIsNone(recovery_v3_progress(repo,1000))
                save('AUDIT.json',dict(input_identity=ident,errors=[],records=28512,updates=36864,new_updates=18432))
                result=recovery_v3_progress(repo,9999);self.assertEqual(result['phase'],'complete');self.assertTrue(result['recovery']['audited']);self.assertFalse(result['learned_advantage'])

    def test_night_waiting_is_not_compute_or_public_process_data(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/overnight_20260917_v1';root.mkdir(parents=True)
            code='source';(root/'supervise_v2.py').write_text(code);sha=hashlib.sha256(code.encode()).hexdigest()
            cfg=dict(source_sha256=sha);state=dict(source_sha256=sha,phase='waiting_resources',observed_unix=1000.,started_unix=900.,active_s=0.,
                window_start_unix=1789579473,window_end_unix=1789608273,registered=['a39_recovery_owner'],current='a39_recovery_owner',error=None,supervisor=dict(private='/home/PRIVATE'))
            (root/'SUPERVISOR_V2.json').write_text(json.dumps(cfg));(root/'STATUS_V2.json').write_text(json.dumps(state))
            with patch('build.joint_supervisor_alive',return_value=True):
                out=night_recovery_progress(repo,1000);self.assertEqual(out['phase'],'waiting_resources');self.assertFalse(out['waiting_is_compute']);self.assertFalse(out['learned_advantage']);self.assertNotIn('PRIVATE',json.dumps(out))
                self.assertEqual(night_recovery_progress(repo,1121)['phase'],'stale')
            (root/'supervise_v2.py').write_text('changed');self.assertIsNone(night_recovery_progress(repo,1000))

    def test_coverage_screen_private_stale_counts_terminal_audit(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/coverage_literal_20260917_v1';root.mkdir(parents=True)
            def save(name,obj):(root/name).write_text(json.dumps(obj))
            labels=[f'{p}_{size}_{seed}' for p in ('tc_mixed','lit_mixed','lit_raw') for size in (384,1536) for seed in (42,43,44)]
            rows=[dict(n=n,split=split) for split,k in [('train',512),('validation',48),('holdout',48),('development',48),('old_validation',24)] for n in (200,300,350) for _ in range(k)]
            for i,r in enumerate(rows):r['index']=i
            ident=dict(kind='bundle_digest',value='8'*64,scope='fresh_coverage_inputs')
            cfg=dict(rows=rows,lanes=[dict(label=l) for l in labels],steps=6144,total_updates=110592)
            state=dict(input_identity=ident,phase='running',stage='training_wave0',observed_unix=1000.,elapsed_s=30.,window_end_unix=1789608273,
                training=[dict(label=labels[0],updates=64,target_updates=6144,phase='training',private='PRIVATE')],completed_updates=64,target_updates=110592,
                teacher_records=1680,target_teacher_records=1680,receipts=[],supervisor=dict(private='PRIVATE'))
            save('CONFIG.json',cfg);save('FROZEN.json',dict(input_identity=ident));save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True):
                out=coverage_screen_progress(repo,1000);self.assertEqual(out['completed_updates'],64);self.assertFalse(out['learned_advantage']);self.assertNotIn('PRIVATE',json.dumps(out))
                self.assertEqual(coverage_screen_progress(repo,1121)['phase'],'stale')
            state['completed_updates']=128;save('STATUS.json',state);self.assertIsNone(coverage_screen_progress(repo,1000))
            state.update(phase='complete',stage='awaiting_independent_audit',completed_updates=110592,cleanup=dict(errors=[],unreaped=[]),
                training=[dict(label=l,phase='complete',updates=6144,target_updates=6144) for l in labels],
                receipts=[dict(label=l,returncode=0) for l in labels+[f'teacher{i}' for i in range(6)]+['targets']])
            save('STATUS.json',state);save('FINAL.json',state)
            self.assertFalse(coverage_screen_progress(repo,9999)['audited'])
            save('AUDIT.json',dict(input_identity=ident,errors=[],updates=110592));self.assertTrue(coverage_screen_progress(repo,9999)['audited'])
            state['receipts'][0]['returncode']=1;save('STATUS.json',state);save('FINAL.json',state);self.assertIsNone(coverage_screen_progress(repo,1000))
            state.update(phase='failed',error='/home/PRIVATE');save('STATUS.json',state);save('FINAL.json',state)
            self.assertEqual(coverage_screen_progress(repo,1000)['phase'],'failed')

    def test_boundary_screen_reused_private_stale_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/boundary_factorial_20260916_v1';root.mkdir(parents=True)
            def save(name,obj):(root/name).write_text(json.dumps(obj))
            labels=[f'd{depth}_{loss}{seed}' for depth in (3,6) for loss in ('base','boundary') for seed in (42,43,44)]
            rows=[dict(n=n,split=s) for s,k in [('train',128),('validation',24)] for n in (200,300,350) for _ in range(k)]
            for i,r in enumerate(rows):r['index']=i
            ident=dict(kind='bundle_digest',value='e'*64,scope='boundary_screen_inputs')
            cfg=dict(lanes=[dict(label=l) for l in labels],gpus=list(range(6)),rows=rows,target_updates=13824,updates_per_lane=1152,epochs=24,batch_size=8)
            state=dict(phase='running',stage='wave0',input_identity=ident,observed_unix=1100.,started_unix=1000.,elapsed_s=100.,
                       training=[dict(label=labels[0],updates=12,target_updates=1152,epoch=1,phase='training',private='PRIVATE')],
                       completed_updates=12,target_updates=13824,receipts=[],supervisor=dict(private='PRIVATE'))
            save('FROZEN.json',dict(input_identity=ident));save('CONFIG.json',cfg);save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True):
                out=boundary_screen_progress(repo,1100);self.assertEqual(out['completed_updates'],12)
                self.assertNotIn('PRIVATE',json.dumps(out));self.assertFalse(out['learned_advantage']);self.assertEqual(out['solver_cells'],0)
                self.assertEqual(boundary_screen_progress(repo,1221)['phase'],'stale')
            with patch('build.joint_supervisor_alive',return_value=False):self.assertEqual(boundary_screen_progress(repo,1100)['phase'],'stale')
            state['completed_updates']=24;save('STATUS.json',state);self.assertIsNone(boundary_screen_progress(repo,1100))
            state.update(phase='complete',stage='complete',completed_updates=13824,cleanup=dict(errors=[],unreaped=[]))
            state['training']=[dict(label=l,phase='complete',updates=1152,target_updates=1152,epoch=24) for l in labels]
            save('STATUS.json',state);save('FINAL.json',state);self.assertIsNone(boundary_screen_progress(repo,1100))
            state['receipts']=[dict(label=l,returncode=0) for l in labels];save('STATUS.json',state);save('FINAL.json',state)
            self.assertEqual(boundary_screen_progress(repo,9999)['phase'],'complete')
            self.assertEqual(boundary_screen_progress(repo,9999)['performance_verdict'],'reused_validation_only')
            state['receipts'][0]['returncode']=1;save('STATUS.json',state);save('FINAL.json',state);self.assertIsNone(boundary_screen_progress(repo,1100))
            state.update(phase='failed',stage='wave0',error='/home/PRIVATE');save('STATUS.json',state);save('FINAL.json',state)
            self.assertEqual(boundary_screen_progress(repo,1100)['phase'],'failed')

    def test_compact_distill_live_stale_private_and_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/compact_distill_20260916_v1';root.mkdir(parents=True)
            def save(name,obj):(root/name).write_text(json.dumps(obj))
            ident=dict(kind='bundle_digest',value='c'*64,scope='distillation_inputs')
            rows=[dict(n=n,split=s) for s,k in [('train',128),('validation',24),('development',24)] for n in (200,300,350) for _ in range(k)]
            for i,r in enumerate(rows):r['index']=i
            labels=[f'{f}{s}' for s in (42,43,44) for f in ('cap','scratch')]
            cfg=dict(target_cells=1440,updates_per_lane=1152,epochs=24,depth=3,batch_size=8,
                     cores=list(range(24)),gpus=list(range(6)),rows=rows,lanes=[dict(label=l) for l in labels])
            state=dict(phase='running',stage='training',completed_cells=0,target_cells=1440,input_identity=ident,
                       started_unix=1000.,observed_unix=1100.,elapsed_s=100.,receipts=[],supervisor={'private':'PRIVATE'},
                       training=[dict(label=l,updates=12,epoch=1,phase='training',target_updates=1152,private='PRIVATE') for l in labels])
            save('FROZEN.json',dict(input_identity=ident));save('CONFIG.json',cfg);save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True) as alive:
                out=compact_distill_progress(repo,1100);self.assertEqual(out['training_updates'],72)
                alive.assert_called_with(repo,root,state,'owner.py')
                self.assertNotIn('PRIVATE',json.dumps(out));self.assertFalse(out['learned_advantage'])
                self.assertEqual(compact_distill_progress(repo,1221)['phase'],'stale')
            with patch('build.joint_supervisor_alive',return_value=False):
                self.assertEqual(compact_distill_progress(repo,1100)['phase'],'stale')
            state['training'].append(state['training'][0]);save('STATUS.json',state)
            self.assertIsNone(compact_distill_progress(repo,1100));state['training'].pop()
            state.update(phase='complete',stage='complete',completed_cells=1440,cleanup=dict(errors=[],unreaped=[]))
            for lane in state['training']:lane.update(phase='complete',updates=1152,epoch=24)
            save('STATUS.json',state);save('FINAL.json',state)
            self.assertIsNone(compact_distill_progress(repo,1100))
            state['receipts']=[dict(label=l,returncode=0) for l in ['teacher_train','teacher_dev','references','evaluate']+[f'{s}{i}' for s in ('train','predict') for i in range(6)]]
            save('STATUS.json',state);save('FINAL.json',state)
            out=compact_distill_progress(repo,9999);self.assertEqual(out['phase'],'complete');self.assertEqual(out['training_updates'],6912)
            self.assertEqual(out['performance_verdict'],'not_evaluated_by_progress_collector')
            state['receipts'][0]['returncode']=1;save('STATUS.json',state);save('FINAL.json',state)
            self.assertIsNone(compact_distill_progress(repo,1100))
            state.update(phase='failed',stage='training',error='/home/PRIVATE');save('STATUS.json',state);save('FINAL.json',state)
            self.assertEqual(compact_distill_progress(repo,1100)['phase'],'failed')
            state['elapsed_s']=float('nan');save('STATUS.json',state);save('FINAL.json',state)
            self.assertIsNone(compact_distill_progress(repo,1100))

    def test_local_feedback_fail_closed_and_private(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/local_feedback_20260916_v1';root.mkdir(parents=True)
            def save(name,obj): (root/name).write_text(json.dumps(obj))
            identity=dict(kind='bundle_digest',value='b'*64,scope='calibration_inputs')
            cfg=dict(target_cells=2448,budgets=[.75,3.,6.],families=['full','random8','random16','cap16'],cores=list(range(24)),gpus=[3,5],rows=[dict(index=i,n=[200,300,350][i//8]) for i in range(24)])
            state=dict(input_identity=identity,phase='running',stage='feedback',completed_cells=34,target_cells=2448,
                       started_unix=1000.,observed_unix=1100.,elapsed_s=100.,receipts=[],supervisor={'private':'PRIVATE'})
            save('FROZEN.json',dict(input_identity=identity));save('CONFIG.json',cfg);save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True):
                out=local_feedback_progress(repo,1100);self.assertEqual(out['completed_cells'],34)
                self.assertNotIn('PRIVATE',json.dumps(out));self.assertFalse(out['learned_advantage'])
                self.assertEqual(local_feedback_progress(repo,1221)['phase'],'stale')
            state.update(phase='complete',stage='complete',completed_cells=2448,cleanup=dict(errors=[],unreaped=[]))
            save('STATUS.json',state);save('FINAL.json',state)
            self.assertIsNone(local_feedback_progress(repo,1100))
            state['receipts']=[dict(label=x,returncode=0) for x in ['reference','predict_rlaf','predict_cap','feedback']]
            save('STATUS.json',state);save('FINAL.json',state)
            self.assertEqual(local_feedback_progress(repo,1100)['phase'],'complete')
            state['cleanup']['unreaped']=[{'private':'PRIVATE'}];save('STATUS.json',state);save('FINAL.json',state)
            self.assertIsNone(local_feedback_progress(repo,1100))

    def test_joint_feedback_live_stale_private_and_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/joint_feedback_20260916_v1';root.mkdir(parents=True)
            def save(name,obj):(root/name).write_text(json.dumps(obj))
            identity=dict(kind='bundle_digest',value='a'*64,scope='evaluation_inputs')
            cfg=dict(input_identity=identity,seeds=[42,43,44],iterations=48,training_cells=27648,
                development_cells=1584,cores=list(range(24)),gpus=[0,3,7])
            state=dict(input_identity=identity,phase='running',stage='training',started_unix=1000.,observed_unix=1100.,elapsed_s=100.,
                supervisor={'pid':1,'private':'PRIVATE'},training=[dict(seed=s,iteration=1,completed_cells=192,training_target=9216,phase='training',private='PRIVATE') for s in [42,43,44]],
                development_completed_cells=0,development_target_cells=1584,running=[{'private':'PRIVATE'}],receipts=[])
            save('FROZEN.json',cfg);save('STATUS.json',state)
            with patch('build.joint_supervisor_alive',return_value=True):
                out=joint_feedback_progress(repo,1100);self.assertEqual(out['phase'],'running')
                self.assertEqual(out['training_completed_cells'],576)
                self.assertNotIn('PRIVATE',json.dumps(out));self.assertNotIn('supervisor',out)
                self.assertFalse(out['learned_advantage']);self.assertEqual(out['performance_verdict'],'not_evaluated_by_progress_collector')
                self.assertEqual(joint_feedback_progress(repo,1221)['phase'],'stale')
            with patch('build.joint_supervisor_alive',return_value=False):
                self.assertEqual(joint_feedback_progress(repo,1100)['phase'],'stale')
            state['training'][0]['completed_cells']=193;save('STATUS.json',state)
            self.assertIsNone(joint_feedback_progress(repo,1100))
            state['training'][0]['completed_cells']=192
            state.update(phase='complete',stage='complete',running=[])
            save('STATUS.json',state);self.assertIsNone(joint_feedback_progress(repo,1100))
            save('FINAL.json',state);save('DEVELOPMENT_COMPLETE.json',dict(cells=1584,formulas=72))
            self.assertIsNone(joint_feedback_progress(repo,1100))
            for lane in state['training']:lane.update(iteration=48,completed_cells=9216,phase='complete')
            state.update(development_completed_cells=1584,receipts=[dict(label=x,returncode=0) for x in
                ['references','feature0','feature1','feature2','train0','train1','train2','predict0','predict1','predict2','predict_rlaf','development']])
            save('STATUS.json',state);save('FINAL.json',state)
            out=joint_feedback_progress(repo,100000);self.assertEqual(out['phase'],'complete')
            self.assertFalse(out['learned_advantage']);self.assertEqual(out['performance_verdict'],'not_evaluated_by_progress_collector')
            state['receipts'][0]['returncode']=1;save('STATUS.json',state);save('FINAL.json',state)
            self.assertIsNone(joint_feedback_progress(repo,1100))
            state.update(phase='failed',error='/home/PRIVATE',stage='training')
            save('STATUS.json',state);save('FINAL.json',state)
            self.assertEqual(joint_feedback_progress(repo,1100)['phase'],'failed')
            state['elapsed_s']=float('nan');save('STATUS.json',state);save('FINAL.json',state)
            self.assertIsNone(joint_feedback_progress(repo,1100))

    def test_prefix_campaign_stale_terminal_and_public_allowlist(self):
        import datetime as dt
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/shared_prefix_20260916_v3';root.mkdir(parents=True)
            def save(name,obj):
                body=json.dumps(obj).encode();(root/name).write_bytes(body);return hashlib.sha256(body).hexdigest()
            digest=save('FROZEN.json',dict(duration_s=28800,cpus=[49,68],private='PRIVATE'))
            state=dict(frozen_sha256=digest,duration_s=28800,cpus=[49,68],started_at='2026-09-16T02:00:00+08:00',
                observed='2026-09-16T02:01:00+08:00',planned_end='2026-09-16T10:00:00+08:00',
                elapsed_s=60,active_coverage_s=55,phase='running',completed=[],failed=[],running=[{'private':'PRIVATE'}],
                completed_jobs=0,failed_jobs=0,completed_cells=0,supervisor_pid=1,supervisor_create_time=0)
            save('STATUS.json',state);now=dt.datetime.fromisoformat(state['observed'])
            with patch('build.prefix_supervisor_alive',return_value=True):
                out=prefix_campaign_progress(repo,now);self.assertEqual(out['phase'],'running')
                self.assertNotIn('PRIVATE',json.dumps(out));self.assertNotIn('supervisor_pid',out)
                self.assertFalse(out['duration_target_met']);self.assertEqual(out['performance_verdict'],'pending')
                self.assertEqual(prefix_campaign_progress(repo,now+dt.timedelta(minutes=3))['phase'],'stale')
            with patch('build.prefix_supervisor_alive',return_value=False):
                self.assertEqual(prefix_campaign_progress(repo,now)['phase'],'stale')
            state.update(phase='window_complete',duration_target_met=True)
            save('STATUS.json',state);save('FINAL.json',state)
            self.assertIsNone(prefix_campaign_progress(repo,now))
            state.update(elapsed_s=28800,observed=state['planned_end'],running=[])
            save('STATUS.json',state);save('FINAL.json',state)
            self.assertTrue(prefix_campaign_progress(repo)['duration_target_met'])
            state['active_coverage_s']=float('nan');save('STATUS.json',state);save('FINAL.json',state)
            self.assertIsNone(prefix_campaign_progress(repo))

    def test_recognition_repair_binding_gate_and_public_allowlist(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/recognition_20260915_v1';root.mkdir(parents=True)
            def save(name,obj):
                b=json.dumps(obj).encode();(root/name).write_bytes(b);return hashlib.sha256(b).hexdigest()
            audit=dict(verdict='full_record_replay_pass',cells=4608,graphs=64,source_frozen=True)
            for name,key in [('FROZEN.json','frozen'),('raw.jsonl.gz','raw'),('charged.jsonl.gz','charged')]:
                audit[key+'_sha256']=save(name,{'private':'PRIVATE'})
            save('STATUS.json',dict(phase='terminal',completed_cells=4608))
            raw=dict(cells=4608,graphs=64,repeats=3,learned_advantage=False,all_gates_pass=True,
                     frozen_sha256=audit['frozen_sha256'],summaries={},by_scale={},contrasts={},gates={})
            def stats(count,ms):
                return dict(count=count,solved=count,recognized=count,par2_ms=ms,median_ms=ms,std_ms=.1,p95_ms=ms,
                    stages_ms={k:.1 for k in ['startup_s','spawn_s','readiness_s','request_s','python_audit_s','exit_s']},private='PRIVATE')
            for form in ['canonical','reversed','shuffled']:
                for key in ['summaries','by_scale','contrasts','gates']:raw[key][form]={}
                for reg in ['cold','resident']:
                    def group(count):return {a:stats(count,5 if a=='new_exact' else 10) for a in ['old_exact','new_exact','new_witness','new_degree32']}
                    raw['summaries'][form][reg]=group(192)
                    raw['by_scale'][form][reg]={n:group(48) for n in ['24','32','48','64']}
                    raw['contrasts'][form][reg]=dict(gain_ms=5,lower_ms=4,upper_ms=6)
                    raw['gates'][form][reg]=True
            def refresh():
                audit['results_sha256']=save('RESULTS.json',raw);audit['gates']=raw['gates'];save('AUDIT.json',audit)
            refresh();out=recognition_repair_progress(repo)
            self.assertEqual(out['cells'],4608);self.assertTrue(out['all_gates_pass']);self.assertNotIn('PRIVATE',json.dumps(out))
            raw['gates']['shuffled']['resident']=False;refresh();self.assertIsNone(recognition_repair_progress(repo))
            raw['gates']['shuffled']['resident']=True
            raw['contrasts']['canonical']['cold']['lower_ms']=-1;refresh();self.assertIsNone(recognition_repair_progress(repo))
            raw['gates']['canonical']['cold']=False;raw['all_gates_pass']=False;refresh()
            self.assertFalse(recognition_repair_progress(repo)['all_gates_pass'])
            raw['summaries']['canonical']['cold']['new_exact']['par2_ms']=float('nan');refresh();self.assertIsNone(recognition_repair_progress(repo))
            raw['summaries']['canonical']['cold']['new_exact']['par2_ms']=5
            raw['by_scale']['reversed']['cold']['24']['old_exact']['count']=47;refresh();self.assertIsNone(recognition_repair_progress(repo))
            raw['by_scale']['reversed']['cold']['24']['old_exact']['count']=48
            raw['learned_advantage']=True;refresh();self.assertIsNone(recognition_repair_progress(repo))
            raw['learned_advantage']=False;refresh();(root/'raw.jsonl.gz').write_bytes(b'drift');self.assertIsNone(recognition_repair_progress(repo))

    def test_confirmation_export_keeps_panels_and_rejects_claim_drift(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/confirm_20260915_e65';root.mkdir(parents=True)
            def save(name,obj):(root/name).write_text(json.dumps(obj))
            identity=dict(kind='bundle_digest',value='a'*64,scope='evaluation_inputs')
            raw=dict(cells=19968,learned_advantage=False,input_identity=identity,totals={},contrasts={},
                confirmation_gates=dict(cold=True,resident=True,private='PRIVATE'),unsat_mechanism_gates=dict(cold=True,resident=True),
                fallback_noninferiority_gates=dict(cold=True,resident=True))
            def stats(n,ms):return dict(count=n,solved=n,par2_ms=ms if n else None,median_ms=ms if n else None,stdev_ms=.1 if n else None)
            for panel in ['confirm','fallback','stress']:
                count=1152 if panel=='confirm' else 96
                arms=['stock','degree32','random32','witness32','exact','pair42','pair43','pair44'] if panel=='confirm' else ['stock','degree32','witness32','exact']
                ns=['48','64'] if panel=='stress' else ['24','32','48','64']
                for reg in ['cold','resident']:
                    for arm in arms:
                        ms=5 if arm=='exact' and panel!='fallback' else 10
                        raw['totals'][f'{panel}/{reg}/{arm}']=dict(stats(count,ms),truth={t:stats(n,ms) for t,n in [('0',0),('10',count*3//4),('20',count//4)]},
                            scales={n:stats(count//len(ns),ms) for n in ns},recognized=0,graph_unknown=0,routes={'legacy':count},private='PRIVATE')
            for reg in ['cold','resident']:
                for arm in ['stock','degree32','random32','witness32']:
                    raw['contrasts'][f'confirm/{reg}/{arm}']=dict(graphs=128,mean_ms=5,lower_ms=4,upper_ms=6)
                raw['contrasts'][f'unsat/{reg}']=dict(graphs=32,mean_ms=5,lower_ms=4,upper_ms=6)
                raw['contrasts'][f'fallback/{reg}']=dict(graphs=32,mean_ms=0,lower_ms=-.01,upper_ms=.01)
            save('STATUS.json',dict(phase='terminal',input_identity=identity))
            names=['FROZEN.json','POOL.json','RECEIPT.json','RESULTS.json','raw.jsonl.gz','charged.jsonl.gz']
            for name in names:save(name,{'private':'PRIVATE'})
            def refresh():
                save('RESULTS.json',raw);h=hashlib.sha256()
                for name in names:
                    body=(root/name).read_bytes();label=name.encode()
                    h.update(len(label).to_bytes(8,'big'));h.update(label);h.update(len(body).to_bytes(8,'big'));h.update(body)
                save('AUDIT.json',dict(verdict='full_replay_pass',cells=19968,input_identity=identity,
                    export_identity=dict(kind='bundle_digest',value=h.hexdigest(),scope='release')))
            refresh();out=confirmation_progress(repo)
            self.assertEqual(out['cells'],19968);self.assertEqual(len(out['totals']),32)
            self.assertNotIn('PRIVATE',json.dumps(out));self.assertFalse(out['learned_advantage'])
            raw['confirmation_gates']['cold']=False;refresh();self.assertIsNone(confirmation_progress(repo))
            raw['confirmation_gates']['cold']=True
            raw['fallback_noninferiority_gates']['resident']=False;refresh();self.assertIsNone(confirmation_progress(repo))
            raw['fallback_noninferiority_gates']['resident']=True
            raw['totals']['confirm/cold/exact']['par2_ms']=float('nan');refresh();self.assertIsNone(confirmation_progress(repo))
            raw['totals']['confirm/cold/exact']['par2_ms']=5
            raw['learned_advantage']=True;refresh();self.assertIsNone(confirmation_progress(repo))
            raw['learned_advantage']=False;refresh()
            (root/'charged.jsonl.gz').write_bytes(b'changed');self.assertIsNone(confirmation_progress(repo))

    def test_exact_search_hashes_scopes_and_gate_recomputation(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/exact_search_20260915_e64';root.mkdir(parents=True)
            def save(name,obj):
                body=json.dumps(obj).encode();(root/name).write_bytes(body);return hashlib.sha256(body).hexdigest()
            audit={'verdict':'complete_accounting_and_witness_replay_pass','cells':18432}
            for name in ['FROZEN.json','RECEIPT.json','TEST_RECEIPT.json','raw.jsonl.gz','charged.jsonl.gz']:
                audit[name.replace('.','_')+'_sha256']=save(name,{'private':'PRIVATE'})
            save('STATUS.json',dict(phase='terminal',receipt_sha256=audit['RECEIPT_json_sha256']))
            def stats(count,ms):return dict(count=count,solved=count,par2_ms=ms,median_ms=ms,stdev_ms=.1)
            raw=dict(cells=18432,learned_advantage=False,receipt_sha256=audit['RECEIPT_json_sha256'],totals={},contrasts={},
                     engineering_gates=dict(cold=True,resident=True,private='PRIVATE'),unsat_mechanism_gates=dict(cold=True,resident=True))
            for reg in ['cold','resident']:
                for arm in ['stock','degree32','random32','pair42','pair43','pair44','witness32','exact']:
                    ms=5 if arm=='exact' else 10
                    raw['totals'][f'{reg}/{arm}']=dict(stats(1152,ms),scales={str(n):stats(288,ms) for n in [24,32,48,64]},
                        truth={'10':stats(864,ms),'20':stats(288,ms)},private='PRIVATE')
                for control in ['stock','degree32','random32','witness32']:
                    raw['contrasts'][f'{reg}/{control}']=dict(graphs=32 if control=='witness32' else 128,gain_ms=5,simultaneous95_lower_ms=4)
            def refresh():
                audit['RESULTS_json_sha256']=save('RESULTS.json',raw);save('AUDIT.json',audit)
            refresh();result=exact_search_progress(repo)
            self.assertEqual(result['cells'],18432);self.assertNotIn('PRIVATE',json.dumps(result))
            self.assertFalse(result['learned_advantage'])
            raw['engineering_gates']['cold']=False;refresh();self.assertIsNone(exact_search_progress(repo))
            raw['engineering_gates']['cold']=True
            raw['unsat_mechanism_gates']['resident']=False;refresh();self.assertIsNone(exact_search_progress(repo))
            raw['unsat_mechanism_gates']['resident']=True
            raw['totals']['cold/exact']['par2_ms']=float('nan');refresh();self.assertIsNone(exact_search_progress(repo))
            raw['totals']['cold/exact']['par2_ms']=5
            raw['learned_advantage']=True;refresh();self.assertIsNone(exact_search_progress(repo))
            raw['learned_advantage']=False;refresh()
            (root/'raw.jsonl.gz').write_bytes(b'changed');self.assertIsNone(exact_search_progress(repo))

    def test_targeted_readout_requires_hashes_and_rejects_false_learning_gate(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/targeted_20260915_e62';root.mkdir(parents=True)
            def save(name,obj):
                body=json.dumps(obj).encode();(root/name).write_bytes(body);return hashlib.sha256(body).hexdigest()
            fh=save('FROZEN.json',{'cpu':99,'private':'PRIVATE_PATH'})
            rh=save('RECEIPT.json',{'frozen_sha256':fh})
            totals={};contrasts={};stages={}
            for b in ['old','popcnt']:
                for reg in ['cold','resident']:
                    for arm in ['stock','degree32','random32','pair42','pair43','pair44']:
                        key=f'{b}/{reg}/{arm}';ms=7 if b=='old' else 5
                        totals[key]=dict(count=1152,solved=1152,candidates=900,par2_ms=ms,median_ms=ms,stdev_ms=.1,
                            private_weights='PRIVATE',scales={str(n):dict(par2_ms=ms,solved=288) for n in [24,32,48,64]})
                        stages[key]={'features_s':{'mean_us':40 if b=='old' else 20}}
            for reg in ['cold','resident']:
                for arm in ['pair42','pair43','pair44']:
                    contrasts[f'repair/{reg}/{arm}']=dict(gain_ms=2,simultaneous95_lower_ms=1)
                    for c in ['stock','degree32','random32']:
                        contrasts[f'learning/{reg}/{arm}/{c}']=dict(gain_ms=0,simultaneous95_lower_ms=-.1)
            result=dict(cells=27648,totals=totals,contrasts=contrasts,learning_gates={'cold':False,'resident':False},receipt_sha256=rh)
            hh=save('HEADROOM.json',dict(count=4096,additional=10,shards=[dict(validation_count=512,additional=10 if i==0 else 0) for i in range(8)]))
            sh=save('STAGES.json',stages)
            def refresh():
                h=save('RESULTS.json',result)
                save('AUDIT.json',dict(verdict='accounting_and_sampled_witness_replay_pass',cells=27648,
                    results_sha256=h,headroom_sha256=hh,stages_sha256=sh,frozen_sha256=fh,receipt_sha256=rh))
            refresh();out=targeted_progress(repo)
            self.assertEqual(out['headroom']['additional'],10)
            self.assertNotIn('PRIVATE',json.dumps(out));self.assertNotIn('cpu',out)
            result['learning_gates']['cold']=True;refresh();self.assertIsNone(targeted_progress(repo))
            result['learning_gates']['cold']=False
            result['totals']['old/cold/stock']['par2_ms']=float('nan');refresh();self.assertIsNone(targeted_progress(repo))
            result['totals']['old/cold/stock']['par2_ms']=7;refresh()
            (root/'RESULTS.json').write_text('{}');self.assertIsNone(targeted_progress(repo))

    def test_overnight_requires_frozen_config_and_excludes_private_process_data(self):
        import datetime as dt
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/overnight_20260915_e61';root.mkdir(parents=True)
            cfg=dict(target_active_s=28800,training_jobs=[{}]*432,cpu_workers=8,
                     gpus=[{'uuid':'PRIVATE_GPU'}]*3,hashes={'PRIVATE_PATH':'PRIVATE_HASH'})
            raw=json.dumps(cfg).encode();(root/'CAMPAIGN.json').write_bytes(raw)
            state=dict(frozen_sha256=hashlib.sha256(raw).hexdigest(),phase='running',
                started_at='2026-09-15T00:00:00+00:00',observed='2026-09-15T00:01:00+00:00',active_s=50,
                completed=[],failed=[],running=[dict(kind='train',pid=999,job_path='PRIVATE_JOB')])
            def save(): (root/'STATUS.json').write_text(json.dumps(state))
            save();now=dt.datetime.fromisoformat(state['observed'])
            out=overnight_progress(repo,now)
            self.assertEqual(out['phase'],'running');self.assertEqual(out['running']['train'],1)
            for word in ['PRIVATE','pid','uuid','job_path']:self.assertNotIn(word,json.dumps(out))
            self.assertEqual(overnight_progress(repo,now+dt.timedelta(minutes=3))['phase'],'stale')
            state['phase']='complete';save();self.assertIsNone(overnight_progress(repo,now))
            state['phase']='running';state['active_s']=999;save();self.assertIsNone(overnight_progress(repo,now))
            state['active_s']=50;state['frozen_sha256']='wrong';save();self.assertIsNone(overnight_progress(repo,now))

    def test_overnight_counts_only_hash_bound_terminal_job_receipts(self):
        import datetime as dt
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/overnight_20260915_e61';folder=root/'jobs/cpu-1/attempt-1';folder.mkdir(parents=True)
            cfg=dict(target_active_s=28800,training_jobs=[],cpu_workers=8,gpus=[{}]*3)
            frozen=json.dumps(cfg).encode();(root/'CAMPAIGN.json').write_bytes(frozen)
            body=json.dumps(dict(phase='terminal',kind='cpu',cells=1152)).encode();(folder/'RESULTS.json').write_bytes(body)
            receipt=dict(id='cpu-1',kind='cpu',job_path='jobs/cpu-1/attempt-1/JOB.json',result_sha256=hashlib.sha256(body).hexdigest())
            state=dict(phase='running',frozen_sha256=hashlib.sha256(frozen).hexdigest(),active_s=50,
                started_at='2026-09-15T00:00:00+00:00',observed='2026-09-15T00:01:00+00:00',
                completed=[receipt],running=[],failed=[])
            (root/'STATUS.json').write_text(json.dumps(state));now=dt.datetime.fromisoformat(state['observed'])
            self.assertEqual(overnight_progress(repo,now)['counts']['cells'],1152)
            (folder/'RESULTS.json').write_text('{}');self.assertIsNone(overnight_progress(repo,now))

    def test_stability_selection_requires_all_seeds_scales_and_safe_aggregates(self):
        import math
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/stability_20260913_e37';root.mkdir(parents=True)
            def save(name,value):
                raw=json.dumps(value).encode();(root/name).write_bytes(raw);return hashlib.sha256(raw).hexdigest()
            fh=save('READOUT_FROZEN.json',{'private':'/private/experiment'})
            def base(n):return dict(count=n,baseline_covers=n//2,oracle_covers=n,variable_graphs=n,baseline_mean=6,oracle_mean=8)
            results={}
            for c in ['ce_mlp','pair_mlp','ce_graph','pair_graph']:
                passed=c=='pair_mlp';params=241 if c.endswith('mlp') else 4641;mean=6.1 if passed else 6;covers=132 if passed else 128
                workers={s:dict(count=256,steps=600,parameters=params,selected_step=25,mean_q=mean,covers=covers,passed=passed,
                    cpu_choice_disagreements=0,cpu_utility_disagreements=0,choices=[0]*256,checkpoint='/private/file',
                    by_scale={str(n):dict(count=64,mean_q=mean,covers=covers//4) for n in [24,32,48,64]}) for s in ['42','43','44']}
                results[c]=dict(parameters=params,workers=workers,passed=passed,selection_score=mean-6-.001*math.log2(params/241))
            r=dict(scope='factorial_selected_validation_not_solver_performance',frozen_sha256=fh,baseline=dict(summary=base(256),by_scale={str(n):base(64) for n in [24,32,48,64]}),
                results=results,predictions=dict(H1=True,H2=False),selected='pair_mlp',resources={s:dict(samples=20,process_peak_mib=612,pid=123) for s in ['42','43','44']},
                diagnostic={'initial':{'tie_gradient_fraction':.9}},training_elapsed_s=45,dev_generated=False,test_generated=False)
            def save_result():
                h=save('READOUT.json',r);save('AUDIT.json',dict(verdict='integrity_pass',readout_sha256=h,frozen_sha256=fh,cnfs=1280,reference_choices=53760,cpu_checkpoint_choices=3072,
                    validation_history_rechecks=73728,selected=r['selected'],dev_generated=False,test_generated=False))
            self.assertIsNone(stability_progress(repo));save_result();out=stability_progress(repo);self.assertEqual(out['selected'],'pair_mlp')
            for word in ['private','checkpoint','choices','pid']:self.assertNotIn('"'+word+'"',json.dumps(out))
            r['selected']='ce_graph';save_result();self.assertIsNone(stability_progress(repo));r['selected']='pair_mlp'
            w=results['pair_mlp']['workers']['43'];w['passed']=False;save_result();self.assertIsNone(stability_progress(repo));w['passed']=True
            w['by_scale']['24']['covers']=32;save_result();self.assertIsNone(stability_progress(repo));w['by_scale']['24']['covers']=33
            w['mean_q']=float('nan');save_result();self.assertIsNone(stability_progress(repo));w['mean_q']=6.1
            w['cpu_utility_disagreements']=1;save_result();self.assertIsNone(stability_progress(repo));w['cpu_utility_disagreements']=0;save_result()
            (root/'test').mkdir();self.assertIsNone(stability_progress(repo));(root/'test').rmdir()
            (root/'READOUT_FROZEN.json').write_text('{}');self.assertIsNone(stability_progress(repo))

    def test_decoder_utility_validation_gate_privacy_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/decoder_utility_20260913_e36';root.mkdir(parents=True)
            def save(name,value):
                raw=json.dumps(value).encode();(root/name).write_bytes(raw);return hashlib.sha256(raw).hexdigest()
            h=save('READOUT_FROZEN.json',{'private_path':'/private/models'})
            teacher={split:dict(count=n,variable_graphs=n,degree_covers=n//2,oracle_covers=n,degree_mean=6,oracle_mean=8) for split,n in [('train',256),('validation',64)]}
            workers={s:dict(seed=int(s),steps=600,selected_step=0,mean_q=6,covers=32,changed_choices=0,samples=4,process_peak_mib=534,elapsed_s=11,
                choices=[1]*64,checkpoint='/private/weights',by_scale={str(n):dict(count=16,baseline_mean=6,mean_q=6,baseline_covers=8,covers=8) for n in [24,32,48,64]}) for s in ['42','43','44']}
            r=dict(scope='selected_validation_not_solver_performance',frozen_sha256=h,teacher=teacher,workers=workers,training_elapsed_s=14,
                diagnostic={'validation':{a:dict(greedy_covers=18,deployed_covers=21) for a in ['untrained','tiny42','tiny43','tiny44']}},label_gate=True,training_gate=False,dev_generated=False,test_generated=False)
            def save_result():
                rh=save('READOUT.json',r);save('AUDIT.json',dict(verdict='integrity_pass',readout_sha256=rh,frozen_sha256=h,cnfs=320,diagnostic_rows=160,cpu_checkpoint_replays=192,
                    label_gate=r['label_gate'],training_gate=r['training_gate'],dev_generated=False,test_generated=False))
            self.assertIsNone(decoder_utility_progress(repo));save_result();out=decoder_utility_progress(repo)
            self.assertFalse(out['training_gate']);self.assertEqual(out['workers']['43']['selected_step'],0)
            for word in ['private','checkpoint','choices','path']:self.assertNotIn('"'+word+'"',json.dumps(out))
            for key in ['training_gate','label_gate']:
                r[key]=not r[key];save_result();self.assertIsNone(decoder_utility_progress(repo));r[key]=not r[key]
            workers['42']['by_scale']['64']['covers']=9;save_result();self.assertIsNone(decoder_utility_progress(repo));workers['42']['by_scale']['64']['covers']=8
            workers['42']['mean_q']=float('nan');save_result();self.assertIsNone(decoder_utility_progress(repo));workers['42']['mean_q']=6;save_result()
            (root/'dev').mkdir();self.assertIsNone(decoder_utility_progress(repo));(root/'dev').rmdir()
            (root/'READOUT.json').write_text('{}');self.assertIsNone(decoder_utility_progress(repo))

    def test_resident_audits_separate_conversion_from_learning_and_omit_private_data(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/resident_policy_20260913_e35';(root/'dev').mkdir(parents=True)
            def save(name,value):
                raw=json.dumps(value).encode();(root/name).write_bytes(raw);return hashlib.sha256(raw).hexdigest()
            bh=save('BENCHMARK_FROZEN.json',{'private':'/private/checkpoint'})
            summaries={k+'_'+m:dict(mean=.1 if k!='resident' else .01,median=.1 if k!='resident' else .01,std=.001,p95=.12,count=80) for k in ['old','cold','resident'] for m in ['stock','tiny42']}
            b=dict(summaries=summaries,conversion_pass=True,frozen_sha256=bh,resident_setup_s=.01,cpu=123,weights='/private/weights')
            br=save('BENCHMARK.json',b)
            save('BENCHMARK_AUDIT.json',dict(verdict='conversion_integrity_pass_not_learning_advantage',benchmark_sha256=br,frozen_sha256=bh,cells=480,independent_formulas=8,conversion_pass=True))
            self.assertTrue(resident_progress(repo)['benchmark']['conversion_pass'])
            arms=['stock','random','degree','static','untrained','tiny42','tiny43','tiny44']
            h=save('dev/FROZEN.json',dict(rows=[{}]*64,arms=arms,seeds=[42,43,44]))
            save('dev/STATUS.json',dict(created='2026-09-13T12:00:00Z',phase='terminal',completed_cells=1536,target_cells=1536,frozen_sha256=h,worker_cpus=[1,2]))
            self.assertIsNone(resident_progress(repo)['development']['readout'])
            def sm(n):return {a:dict(solved=n,sat=n,unsat=0,candidates=0,par2_s=.01,full_s={'median':.01},private='/private/data') for a in arms}
            r=dict(frozen_sha256=h,cells=1536,trials=192,instances=64,engineering_pass=False,learning_pass=False,summaries=sm(192),by_scale={str(n):sm(48) for n in [24,32,48,64]})
            rh=save('dev/RESULTS.json',r);ch=save('SELECTION.json',dict(engineering=False,learning=False,results_sha256=rh))
            save('dev/AUDIT.json',dict(verdict='integrity_pass',frozen_sha256=h,results_sha256=rh,selection_sha256=ch,engineering=False,learning=False,test_generated=False))
            result=resident_progress(repo);self.assertEqual(result['development']['phase'],'terminal');self.assertTrue(result['development']['readout']['no_test'])
            for word in ['private','weights','worker_cpus','checkpoint']:self.assertNotIn(word,json.dumps(result))
            (root/'test').mkdir();self.assertIsNone(resident_progress(repo));(root/'test').rmdir()
            r['summaries']['tiny42']['par2_s']=float('nan');save('dev/RESULTS.json',r);self.assertIsNone(resident_progress(repo))

    def test_structured_panels_require_audited_readout_and_exclude_private_fields(self):
        arms=['stock','random','degree','original','untrained','cap42','cap43','cap44']
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/phase_budget_20260913_e33';(root/'dev').mkdir(parents=True)
            cfg=dict(created='2026-09-13T10:00:00Z',rows=[{}]*64,arms=arms,seeds=[42,43,44]);raw=json.dumps(cfg).encode();h=hashlib.sha256(raw).hexdigest()
            (root/'dev/FROZEN.json').write_bytes(raw)
            state=dict(observed=cfg['created'],phase='terminal',completed_cells=1536,target_cells=1536,frozen_sha256=h,worker_cpus=[11,12])
            (root/'dev/STATUS.json').write_text(json.dumps(state));self.assertIsNone(structured_progress(repo)['readout'])
            totals={a:dict(solved=192,sat=153,unsat=39,candidates=100,par2_s=.05,private_weights=[1,2]) for a in arms}
            scales={str(n):{a:dict(solved=48,sat=s,unsat=48-s,candidates=25,par2_s=.05) for a in arms} for n,s in zip([24,32,48,64],[39,38,38,38])}
            r=dict(instances=64,trials=192,cells=1536,errors=[],engineering_pass=False,learning_pass=False,frozen_sha256=h,summaries=totals,by_scale=scales)
            raw=json.dumps(r).encode();(root/'dev/RESULTS.json').write_bytes(raw);rh=hashlib.sha256(raw).hexdigest()
            audit=dict(verdict='integrity_pass_development_gates_failed',results_sha256=rh,frozen_sha256=h,test_generated=False)
            (root/'AUDIT.json').write_text(json.dumps(audit));(root/'SELECTION.json').write_text(json.dumps(dict(engineering=False,learning=False,results_sha256=rh)))
            out=structured_progress(repo);self.assertEqual(out['readout']['summaries']['degree']['par2_s'],.05)
            self.assertNotIn('private_weights',json.dumps(out));self.assertNotIn('worker_cpus',json.dumps(out))
            (root/'test').mkdir();self.assertIsNone(structured_progress(repo))

    def test_gpu_supervisor_requires_all_zero_exits_and_matching_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/shared_gpu_20260913_e32';root.mkdir(parents=True)
            frozen=json.dumps(dict(seeds=[42,43,44],epochs=120,batch_size=16));(root/'FROZEN.json').write_text(frozen);h=hashlib.sha256(frozen.encode()).hexdigest()
            artifact=hashlib.sha256(b'synthetic-checkpoint').hexdigest()
            for seed in [42,43,44]:
                folder=root/f'seed-{seed}';folder.mkdir()
                for name in ['best','last']:(folder/f'{name}.pt').write_bytes(b'synthetic-checkpoint')
                state=dict(phase='terminal',observed='2026-09-13T09:00:00Z',seed=seed,gpu_index=3,epoch=120,frozen_sha256=h,best_sha256=artifact,last_sha256=artifact)
                (folder/'STATUS.json').write_text(json.dumps(state))
            terminal=dict(completed=True,frozen_sha256=h,exit_codes={'42':0,'43':0,'44':0})
            (root/'TERMINAL.json').write_text(json.dumps(terminal));self.assertTrue(shared_gpu_progress(repo)['terminal'])
            replay=root/'resource-replay';replay.mkdir()
            rf=json.dumps(dict(original_frozen_sha256=h)).encode();(replay/'REPLAY_FROZEN.json').write_bytes(rf)
            samples=json.dumps([dict(total_process_mib={'42':928,'43':928,'44':936})]).encode();(replay/'PROCESS_SAMPLES.json').write_bytes(samples)
            rr=dict(completed=True,original_process_memory_still_unmeasured=True,scope='separate_identical_workload_resource_replay',exit_codes={'42':0,'43':0,'44':0},
                    replay_frozen_sha256=hashlib.sha256(rf).hexdigest(),process_samples_sha256=hashlib.sha256(samples).hexdigest(),
                    workers=[dict(seed=s,gpu_index=g,epochs=120,samples=1,sampled_total_process_peak_mib=m) for s,g,m in [(42,3,928),(43,2,928),(44,5,936)]])
            (replay/'RESULT.json').write_text(json.dumps(rr));self.assertEqual(shared_gpu_progress(repo)['resource_replay']['workers']['44']['sampled_total_process_peak_mib'],936)
            rr['workers'][0]['sampled_total_process_peak_mib']=900;(replay/'RESULT.json').write_text(json.dumps(rr));self.assertIsNone(shared_gpu_progress(repo))
            rr['workers'][0]['sampled_total_process_peak_mib']=928;(replay/'RESULT.json').write_text(json.dumps(rr))
            terminal['exit_codes']['44']=1;(root/'TERMINAL.json').write_text(json.dumps(terminal));self.assertIsNone(shared_gpu_progress(repo))
            terminal['exit_codes']['44']=0;(root/'TERMINAL.json').write_text(json.dumps(terminal))
            (root/'seed-44/last.pt').write_bytes(b'changed');self.assertIsNone(shared_gpu_progress(repo))

    def test_gpu_progress_excludes_other_processes_and_rejects_false_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/shared_gpu_20260913_e32';folder=root/'seed-42';folder.mkdir(parents=True)
            frozen=json.dumps(dict(seeds=[42,43,44],epochs=120,batch_size=16));(root/'FROZEN.json').write_text(frozen);h=hashlib.sha256(frozen.encode()).hexdigest()
            state=dict(phase='running',observed='2026-09-13T09:00:00Z',seed=42,gpu_index=3,epoch=12,frozen_sha256=h,pid=999,private_host='private')
            (folder/'STATUS.json').write_text(json.dumps(state));out=shared_gpu_progress(repo)
            self.assertEqual(out['workers']['42']['epoch'],12);self.assertNotIn('private',json.dumps(out));self.assertNotIn('pid',json.dumps(out))
            (root/'GPU_SAMPLES.json').write_text(json.dumps([dict(worker_alive={'42':True},devices=[dict(index=3,utilization=90,uuid='private'),dict(index=6,utilization=1,process='private')])]))
            self.assertEqual(shared_gpu_progress(repo)['telemetry']['42']['card_utilization_mean'],90)
            state.update(phase='terminal',epoch=120);(folder/'STATUS.json').write_text(json.dumps(state));self.assertIsNone(shared_gpu_progress(repo))

    def test_cost_aware_training_units_and_privacy(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/cost_aware_20260913_e30';(root/'train').mkdir(parents=True)
            (root/'train/FROZEN.json').write_text('{}');h=hashlib.sha256(b'{}').hexdigest()
            state=dict(phase='terminal',observed='2026-09-13T09:00:00Z',completed_cells=9216,target_cells=9216,workers=16,frozen_sha256=h,worker_cpus=[1,2])
            (root/'TRAIN_STATUS.json').write_text(json.dumps(state))
            trained=dict(training_cells=9216,train_frozen_sha256=h,selected={a:dict(fitness_s=.3,solved=60,trials=128,weights=[1,2,3,0]) for a in ['cheap','cap']})
            (root/'TRAINED.json').write_text(json.dumps(trained));out=conservative_progress(repo,cost_aware=True)
            self.assertEqual(out['training']['target_cells'],9216)
            self.assertEqual(out['training']['readout']['summaries']['cheap']['fitness_s'],.3)
            self.assertNotIn('fitness_flips',json.dumps(out));self.assertNotIn('weights',json.dumps(out));self.assertNotIn('worker_cpus',json.dumps(out))
            audit=dict(verdict='integrity_pass_candidates_identical_to_hand',trained_sha256=out['training']['readout']['sha256'],training_cells=9216,sat_witnesses_rechecked=2802)
            (root/'AUDIT.json').write_text(json.dumps(audit))
            pre=dict(reason='both_candidates_identical_to_hand',decision='stop_before_development_no_distinct_candidate',dev_generated=False,test_generated=False,protocol_deviation=True,cap_active=False,
                     trained_sha256=audit['trained_sha256'],audit_sha256=hashlib.sha256((root/'AUDIT.json').read_bytes()).hexdigest())
            (root/'PREFLIGHT.json').write_text(json.dumps(pre));self.assertEqual(conservative_progress(repo,cost_aware=True)['preflight']['checked_training_sat'],2802)
            pre['protocol_deviation']=False;(root/'PREFLIGHT.json').write_text(json.dumps(pre));self.assertIsNone(conservative_progress(repo,cost_aware=True))
            pre['protocol_deviation']=True;(root/'PREFLIGHT.json').write_text(json.dumps(pre))
            trained['selected']['cap']['fitness_s']=.6;(root/'TRAINED.json').write_text(json.dumps(trained));self.assertIsNone(conservative_progress(repo,cost_aware=True))

    def test_conservative_dev_trial_denominator_and_confirmation_guard(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/conservative_search_20260913_e29';(root/'train').mkdir(parents=True);folder=root/'dev';folder.mkdir()
            (root/'train/FROZEN.json').write_text('{}');th=hashlib.sha256(b'{}').hexdigest()
            (root/'TRAIN_STATUS.json').write_text(json.dumps(dict(phase='running',observed='2026-09-13T00:00:00Z',completed_cells=0,target_cells=18432,workers=32,frozen_sha256=th)))
            arms=['stock','random','hand','learned_cheap','learned_cap'];frozen=json.dumps(dict(arms=arms));h=hashlib.sha256(frozen.encode()).hexdigest();(folder/'FROZEN.json').write_text(frozen)
            s=dict(phase='terminal',observed='2026-09-13T00:00:00Z',instances=96,completed_rows=288,target_rows=288,completed_cells=1440,target_cells=1440,frozen_sha256=h,prepared_sha256='prep')
            (folder/'STATUS.json').write_text(json.dumps(s))
            def sm(n):return {a:dict(solved=n//2,par2_s=5,sls_solved=n//3) for a in arms}
            r=dict(instances=96,trials=288,cells=1440,frozen_sha256=h,prepared_sha256='prep',confirmed=False,errors=[],summaries=sm(288),by_scale={z:sm(144) for z in ['325','500']})
            (folder/'RESULTS.json').write_text(json.dumps(r));out=conservative_progress(repo)['stages']['dev']
            self.assertEqual(out['instances'],96);self.assertEqual(out['target_rows'],288);self.assertEqual(out['readout']['summaries']['random']['solved'],144)
            r['confirmed']=True;(folder/'RESULTS.json').write_text(json.dumps(r));self.assertIsNone(conservative_progress(repo))
            r['confirmed']=False;r['summaries']['random']['solved']=144.5;(folder/'RESULTS.json').write_text(json.dumps(r));self.assertIsNone(conservative_progress(repo))

    def test_conservative_training_privacy_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/conservative_search_20260913_e29';(root/'train').mkdir(parents=True)
            (root/'train/FROZEN.json').write_text('{}');h=hashlib.sha256(b'{}').hexdigest()
            state=dict(phase='running',observed='2026-09-13T00:00:00Z',completed_cells=1024,target_cells=18432,workers=32,frozen_sha256=h,private_host='secret')
            (root/'TRAIN_STATUS.json').write_text(json.dumps(state));out=conservative_progress(repo)
            self.assertEqual(out['training']['completed_cells'],1024);self.assertNotIn('secret',json.dumps(out));self.assertFalse(out['stages'])
            state['phase']='terminal';(root/'TRAIN_STATUS.json').write_text(json.dumps(state));self.assertIsNone(conservative_progress(repo))
            state['completed_cells']=18432;(root/'TRAIN_STATUS.json').write_text(json.dumps(state))
            trained=dict(training_cells=18432,train_frozen_sha256=h,selected={a:dict(fitness=200000,solved=96,trials=192,weights=[1,2,3,0]) for a in ['cheap','cap']})
            (root/'TRAINED.json').write_text(json.dumps(trained));out=conservative_progress(repo)
            self.assertEqual(out['training']['readout']['summaries']['cheap']['solved'],96);self.assertNotIn('weights',json.dumps(out))
            trained['selected']['cap']['fitness']=float('nan');(root/'TRAINED.json').write_text(json.dumps(trained));self.assertIsNone(conservative_progress(repo))

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
        self.assertEqual(len(plan['actions']),54)
        self.assertEqual(sum(g['state']=='部分改善' for g in plan['gaps']),4)
        self.assertEqual(sum(g['state']=='未解决' for g in plan['gaps']),3)
        actions={a['id']:a for a in plan['actions']}
        self.assertIn('2448',actions['A32']['cost'])
        self.assertIn('6912',actions['A33']['cost'])
        self.assertIn('13824',actions['A34']['cost'])
        self.assertIn('机制筛选',actions['A34']['stop'])
        self.assertIn('5472',actions['A35']['status'])
        self.assertIn('2736',actions['A35']['cost'])
        self.assertIn('不优于',actions['A36']['status'])
        self.assertIn('表示预测门通过',actions['A37']['status'])
        self.assertIn('110592',actions['A38']['cost'])
        self.assertIn('非同FLOPs',actions['A38']['method'])
        self.assertIn('26074',actions['A39']['cost'])
        self.assertIn('28512',actions['A39']['status'])
        self.assertIn('保真门通过',actions['A40']['priority'])
        self.assertIn('432',actions['A43']['status'])
        self.assertIn('2016',actions['A44']['cost'])
        self.assertIn('3168',actions['A45']['cost'])
        self.assertIn('18432',actions['A46']['cost'])
        self.assertIn('减半门全失败',actions['A47']['status'])
        self.assertIn('768',actions['A48']['cost']+actions['A48']['status'])
        self.assertIn('576',actions['A49']['cost']+actions['A49']['status'])
        self.assertIn('1.856%',actions['A49']['stop'])
        self.assertIn('停止静态权重插值',actions['A49']['stop'])
        self.assertIn('122',actions['A50']['cost']+actions['A50']['status'])
        self.assertIn('3.98%',actions['A50']['stop'])
        self.assertIn('停止在线回退家族',actions['A50']['stop'])
        self.assertIn('5,160',actions['A51']['status'])
        self.assertIn('0 新增求解',actions['A51']['cost'])
        self.assertIn('175',actions['A51']['stop'])
        self.assertIn('制式属性',actions['A51']['stop'])
        self.assertIn('end_to_end_advantage 标志不变',actions['A51']['stop'])
        self.assertIn('不设成功门',actions['A51']['gate'])
        self.assertIn('2304',actions['A52']['cost']+actions['A52']['status'])
        self.assertIn('3.81%',actions['A52']['stop'])
        self.assertIn('关闭规模规则家族',actions['A52']['stop'])
        self.assertEqual(actions['A52']['depends_on'],['A48','A49'])
        self.assertIn('4,200',actions['A53']['status'])
        self.assertIn('T1=10',actions['A53']['method'])
        self.assertIn('−153.5',actions['A53']['stop'])
        self.assertIn('不事后改 T1',actions['A53']['gate'])
        self.assertEqual(actions['A53']['depends_on'],['A51'])
        self.assertIn('98.64%',actions['A41']['stop'])
        self.assertIn('15120',actions['A42']['status'])
        self.assertIn('无family过98%',actions['A42']['stop'])
        self.assertIn('描述性pilot',actions['A33']['gate'])
        self.assertIn('训练切片描述门',actions['A32']['gate'])
        self.assertIn('0/6过门',actions['A31']['status'])
        self.assertIn('27648',actions['A31']['cost'])
        self.assertIn('多余解析',actions['A31']['stop'])
        self.assertIn('E26',actions['A3']['status'])
        self.assertIn('E27',actions['A6']['status'])
        self.assertIn('E28',actions['A4']['status'])
        self.assertIn('E29',actions['A7']['status'])
        self.assertIn('E30',actions['A8']['status'])
        self.assertIn('E31',actions['A5']['status'])
        self.assertEqual(actions['A1']['status'],'计划中 · 未启动')
        self.assertIn('E35',actions['A2']['status'])
        self.assertIn('E33',actions['A10']['status'])
        self.assertIn('E34',actions['A11']['status'])
        self.assertIn('E35',actions['A12']['status'])
        self.assertEqual(actions['A13']['status'],'E36 局部信号 · 三种子门未过')
        self.assertEqual(actions['A14']['status'],'E37 成对小网络通过训练门')
        self.assertEqual(actions['A15']['status'],'E60 同图同核冷启动快27.4%；模型优势仍未建立')
        self.assertIn('E61',actions['A16']['status'])
        self.assertIn('需另做全成本评估',actions['A16']['gate'])
        self.assertIn('不足8小时',actions['A16']['stop'])
        self.assertIn('学习净优势仍未通过',actions['A17']['status'])
        self.assertIn('非学习优势',actions['A18']['status'])
        self.assertIn('更强整体确认门未通过',actions['A19']['status'])
        self.assertIn('六项修复门通过',actions['A20']['status'])
        self.assertIn('整体门未通过',actions['A21']['status'])
        self.assertIn('仅补全SAT冷启动失败',actions['A21']['gate'])
        self.assertIn('10.667→8.857',actions['A21']['stop'])
        self.assertIn('长尾未解决',actions['A22']['status'])
        self.assertIn('默认冷启动失败',actions['A22']['gate'])
        self.assertIn('8.821→8.315',actions['A22']['stop'])
        self.assertIn('总体6/10',actions['A23']['status'])
        self.assertIn('随机驻留失败',actions['A23']['gate'])
        self.assertIn('6.891→6.096',actions['A23']['stop'])
        self.assertIn('线程启动39.955ms',actions['A23']['stop'])
        self.assertIn('精确驻留降3.07%',actions['A24']['status'])
        self.assertIn('仅补全SAT驻留失败',actions['A24']['gate'])
        self.assertIn('整体6/10',actions['A24']['gate'])
        self.assertIn('2.480→2.404',actions['A24']['stop'])
        self.assertIn('逐原子句检查',actions['A24']['stop'])
        self.assertIn('精确驻留降30.38%',actions['A25']['status'])
        self.assertIn('冷/驻留两门均未通过',actions['A25']['gate'])
        self.assertIn('实现门8/10',actions['A25']['gate'])
        self.assertIn('1.655→1.152',actions['A25']['stop'])
        self.assertIn('慢5.819微秒',actions['A25']['stop'])
        self.assertIn('不删校验',actions['A25']['stop'])
        self.assertIn('困难驻留降6.62%',actions['A27']['status'])
        self.assertIn('15/28',actions['A27']['gate'])
        self.assertIn('1.193481→1.114467',actions['A27']['stop'])
        self.assertIn('17.949微秒',actions['A27']['stop'])
        self.assertIn('不是ExactSearch本身加速',actions['A27']['stop'])
        self.assertIn('匹配精确驻留降19.60%',actions['A28']['status'])
        self.assertIn('10/24',actions['A28']['gate'])
        self.assertIn('1.605263→1.290584',actions['A28']['stop'])
        self.assertIn('退化14.71%',actions['A28']['stop'])
        self.assertIn('未复现20ms长尾',actions['A28']['stop'])
        self.assertIn('0.6478%/0.2722%',actions['A29']['status'])
        self.assertIn('未启动学习训练',actions['A29']['status'])
        self.assertIn('61440',actions['A29']['cost'])
        self.assertIn('两组绝对成本门都失败',actions['A29']['gate'])
        self.assertIn('406.044',actions['A29']['stop'])
        self.assertIn('0/3晋级',actions['A30']['status'])
        self.assertIn('1152',actions['A30']['cost'])
        self.assertIn('376',actions['A30']['gate'])
        self.assertIn('不是96实例',actions['A30']['method'])
        self.assertIn('6/24',actions['A30']['stop'])
        self.assertIn('2/24',actions['A30']['stop'])
        self.assertIn('0.153714',actions['A30']['stop'])
        self.assertIn('区间跨零',actions['A30']['stop'])
        self.assertIn('实际7小时13分36秒',actions['A26']['status'])
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

class CacheAffinityV2Tests(unittest.TestCase):
    def test_audit_identity_denominators_negative_results_and_privacy(self):
        from build import cache_affinity_v2_progress
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);root=repo/'experiments/cache_affinity_20260915_v2/dev';root.mkdir(parents=True)
            frozen=b'{"private_path":"/private/checkpoint"}'
            (root/'FROZEN.json').write_bytes(frozen)
            fh=hashlib.sha256(frozen).hexdigest()
            def sm(cost,cells,hits):
                return dict(par2_ms=cost,cells=cells,solved=cells,timeouts=0,cache_hits=hits,assignment=[1],pid=42)
            r=dict(cells=4608,independent_graphs=128,repeats=3,frozen_sha256=fh,learned_advantage=False,
                source_integrity=True,summaries={},by_scale={},contrasts={},gates={'cold':False,'resident':True},all_lifecycles_pass=False)
            for reg,costs in [('cold',(12,12.1)),('resident',(6,5))]:
                r['summaries'][reg]={arm:sm(cost,1152,768 if reg=='resident' and arm=='cache_random32' else 0) for arm,cost in zip(('random32','cache_random32'),costs)}
                r['by_scale'][reg]={str(n):{arm:sm(cost,288,192 if reg=='resident' and arm=='cache_random32' else 0) for arm,cost in zip(('random32','cache_random32'),costs)} for n in (24,32,48,64)}
                gain=costs[0]-costs[1]
                r['contrasts'][reg]=dict(gain_ms=gain,lower_ms=gain-.2,upper_ms=gain+.2,independent_graphs=128)
            def save():
                raw=json.dumps(r).encode();(root/'RESULTS.json').write_bytes(raw)
                (root/'AUDIT.json').write_text(json.dumps(dict(verdict='integrity_pass',results_sha256=hashlib.sha256(raw).hexdigest(),frozen_sha256=fh,
                    cells=4608,graphs=128,reference_agreements=4608,cache_hits=768,gates=r['gates'])))
            self.assertIsNone(cache_affinity_v2_progress(repo));save()
            out=cache_affinity_v2_progress(repo)
            self.assertTrue(out['gates']['resident']);self.assertFalse(out['gates']['cold'])
            self.assertFalse(out['learning_pass'])
            for word in ('private','assignment','pid','checkpoint'):
                self.assertNotIn(word,json.dumps(out))
            r['gates']['cold']=True;save();self.assertIsNone(cache_affinity_v2_progress(repo));r['gates']['cold']=False
            r['summaries']['resident']['cache_random32']['cache_hits']=767;save();self.assertIsNone(cache_affinity_v2_progress(repo));r['summaries']['resident']['cache_random32']['cache_hits']=768
            r['contrasts']['resident']['lower_ms']=float('nan');save();self.assertIsNone(cache_affinity_v2_progress(repo));r['contrasts']['resident']['lower_ms']=.8
            r['by_scale']['resident']['24']['random32']['par2_ms']=7;save();self.assertIsNone(cache_affinity_v2_progress(repo));r['by_scale']['resident']['24']['random32']['par2_ms']=6
            r['summaries']['cold']['random32']['solved']=1151.5;save();self.assertIsNone(cache_affinity_v2_progress(repo));r['summaries']['cold']['random32']['solved']=1152
            r['learned_advantage']=True;save();self.assertIsNone(cache_affinity_v2_progress(repo));r['learned_advantage']=False;save()
            (root/'RESULTS.json').write_text('{}');self.assertIsNone(cache_affinity_v2_progress(repo));save()
            (root/'FROZEN.json').write_text('{}');self.assertIsNone(cache_affinity_v2_progress(repo))

if __name__=='__main__':
    unittest.main()
