# CAP-SAT Rebuttal Dashboard

Latest: A35 recovery reuses2736records and adds2736, all5472audited; all12models
support in-sample underfit. A36 TC raw-logit loss underperforms mixed; negative
result retained. A37 literal representation mixed improves old-validation
agreement to93.20–94.02%(+8.00–9.50points), raw to95.65–96.28%. This is NOT a
solver or end-to-end result. The zero-update v1 import failure is retained.
A38 completed1680new teacher labels and18models/110592updates on nested384/1536fresh
training formulas,144new validation,144sealed holdout and144sealed development.
See the live A38 panel and plan cards. Six shared GPUs, bounded resources and
window ending2026-09-17 09:24:33Asia/Shanghai; do not claim eight hours complete.
All38016records audited. Literal newvalidation96.58–97.43% vsTC90.38–90.78%;
both representation gates pass but no family reaches absolute98%. A39 learning-
rate comparison stopped at a resource floor:6complete lanes/14256records accepted,
26074loggedupdates including unfinished work. Recoveryv2 reused all6complete
lanes and waited for192MiB root headroom before6remaining lanes;128MiB updatefloor
unchanged, model/Adam/RNG saved on resourcepause. Original failed state retained.
Old supervisorv2 ended at09:24:33 with zero new recovery updates; waiting is not
compute. Approved109MiB HFmodelcache relocation and9duplicate screenshots cleanup
completed; model bytes/original path and identical screenshot copies retained.
A39v3 completed six remaining lanes (18432new+18432reused),28512records audited.
Anneal agreement97.51–97.58%, no family passes98% EACHseed/EACHscale. A40 completed
12depth-width models,110592updates and28512audited records. All four families pass
fidelity, with99.6123–99.7479% exposed-validation agreement. The preregistered
worst-scale/seed choice is d6_raw:99.7278/99.7407/99.7391%. This is teacher-action
imitation, not SAT accuracy or end-to-end advantage; depth AND width change.
A43 audited432prediction requests: teacher18.0249ms versus10.0526/11.2612/11.0761ms,
all seeds/scales pass80%latency and98%fidelity gates. Full singleton prediction
includes read/parse/build/transfer/forward/decode, but no solver and no cold start.
A44 completed1152new predictions and2016full-cost native cells on144previously
unused development formulas. All audited: students improve12.9–15.1% over stock,
but remain0.56–3.21% slower than teacher and4.04–6.78% slower than minimal degree.
All strict feasibility/advantage family gates fail. Full prediction drops from
21.72ms to12.63–13.03ms and fidelity remains99.714–99.749%; search losses persist.
A44 did NOT capture actual-send payload hashes; action/input binding is frozen
code-path review, not independent historical input-receipt replay. A45 separately
captures these receipts while diagnosing phase versus weight errors on ALL144
now-exposed development formulas:3168solver-only oracle-factorial cells. Equal
zero prediction charge isolates actions; it is NOT a deployable speedup benchmark.
Held-out144 remain sealed. A45 completed3168cells with actual-send hashes: oracle
weight repair restores97.714/99.725/89.624% of the three students' excess search
cost; phase repair restores essentially nothing. All weight-attribution gates
pass, all phase gates fail; prediction cost excluded, not deployable speedup.
Prior16:40UTC window ended with all A40–A45 complete. Following the user's new
continue request, A46 is separately frozen: six paired continuations/18432steps,
original versus train-only scale-balanced weight MSE (coefficient53.8409449).
Same source weights/data/schedule, fixed final3072, six sharedGPUs, no teacher at
deployment. All18432updates/14256records now audited: weight RMSE decreases
31.98/37.76/32.64% overall (31.05–38.70% across scales), phase protection passes,
but the prespecified50%reduction gate fails for all seeds. Clipping occurs0times
per original arm and1/2/1times per balanced arm; train/validation errors are close.
No solver or independent-confirmation claim. New bounded end17:20UTC (Sep18
01:20Beijing) was not an old-window extension; this fixed queue already completed.
A47 is newly authorized and frozen: six existing model heads, train-only
formula-balanced float64 residual ridge1e-8, folded into the same architecture.
Actual folded-network predictions determine the precision screen; independent CPU
fit/GPU full-network replay and exact same-hidden phase checks are separate.
CPU/GPU7tests each pass; target21024before/after records,0solver cells. Current
execution follows the identity-checked heartbeat, not this static description.
Absolute end19:30UTC (Sep18 03:30Beijing), no grid expansion, holdout or paper edit.
A47 now complete:21024records/full-network replays audited,14process identities
exited,181.73s. Balanced validation improves another4.63–8.42%, phase flips0;
all50%repair gates still fail. Mechanism MIXED; no general capacity-limit claim.
The output-row expansion stops. A48 separately froze192TRAIN/48streams,
384fresh teacher predictions and768paired full-cost cells to diagnose teacher/
degree complementarity. The first run stopped at97blocks/194cells under a
root-resource floor; recovery waited for192MiB admission and completed the287
missing blocks without changing the scientific identity or128MiB run floor.
All768cells and receipts audit cleanly. The offline formula oracle has only
8.3435% headroom (below the10% gate), while the whole-stream n-only cross-fit
rule improves3.3689% over degree and3.8439% over teacher with no per-scale
harm (size-rule gate passes). Because the dual gate fails and selector cost is
excluded, this is not an end-to-end deployment or CAP advantage;
`may_propose_scale_route=false`. Stop classifier expansion, keep the positive
signal as a hypothesis, and keep the holdout sealed. New stages are not
retroactively counted as the original eight-hour window.
A41 retained-output diagnosis:98.64–99.55% of literal errors are near-zero;
three-seed voting only+.2365/+.3783points, no full-cost inference. A42 fits one
phase threshold per model on training ONLY:raw seeds improve+.7932/+.2870/+.1412
points to97.37–97.52%, mixed is inconsistent and no family passes98%. Independent
NumPy audit replays9training optima/15120records. CPU decoder-only benchmarking
does not establish end-to-end performance. Both analyses use exposed validation;
no GPU updates, new labels or sealed holdout/development evaluation.
No paper rewrite and no fully solved performance gap.

Campaign status: A26 stopped at the disk safety floor after7h13m36s, before its
fixed eight-hour window ended.1172blocks/4,500,480cells were audited; two partial
blocks stopped, no workers remain. The terminal readout timed out and awaits
recovery; it is not a performance pass. The original queue was not restarted.
The live panel verifies heartbeat/process identity and audited completion
receipts; stale exports do not claim execution. No GPU training or paper rewrite.
Historical E61 remains stopped incomplete.

Current mechanism screen: A34 completed3/6layers × baseline/boundary-weighted loss ×
three seeds,12scratch models,13,824updates, on REUSED A33 train/validation only.
The common selection metric and all per-formula outputs are retained. This is
neither new development nor end-to-end evaluation:0solver cells. No automatic
extra epochs, preference fine-tuning or promotion. See `#boundary-screen` and
`#action-A34`; operational completion and scientific gates stay separate.
All12exit receipts and4,320validation records passed audit; queue234.49seconds.
All4mechanism contrasts and4absolute family fidelity screens FAILED. Depth alone
improved near-boundary agreement0.60–0.76percentage points, below the2point gate;
boundary reweighting was inconsistent. Overall agreement89.22–90.05% remains well
below98%. This branch stopped without a solver run. Fitability/train-validation
diagnosis is the next separately specified frontier, not an active training queue.

Previous method pilot: A33 completed and passed raw-record audit, but failed both
family feasibility gates. A compact 3-layer, 166,530-parameter
student distills the fixed RLAF teacher offline, with matched CAP/scratch
initializations and three seeds each. 384 fresh training, 72 validation and 72
development CNFs; 6,912 optimizer updates and 1,440 full-cost solver cells complete.
Training shares six GPUs; singleton prediction timing is sequential on one GPU.
Fidelity, inference cost, solver parity and potential advantage are separate
gates. No automatic preference fine-tuning, paper rewrite or independent claim.
The top `#compact-distill` panel exports operational counts only; audited science
belongs in `#action-A33`. Prediction medians fell43.02–60.64%descriptively, but
phase agreement is only89.54–90.40%; all students lose n350 solves vs teacher.
Validation-only diagnosis localizes89.42–90.67%of phase errors to teacher |rho|<.1.
No automatic extra epochs or preference training. A32 remains failed for expansion.

Earlier method diagnostic: A30 finishes48new general-CNF development cases,
12arms x2fixed-seed timing repeats=1152cells. On n200, cheap degree weights
(.153714s PAR2) outperform native RLAF(.176290s), though both beat stock(.445378s).
On n350, native RLAF solves6/24; replacing its phases with CAP solves2/24.
All three CAP-containing candidates fail promotion. Both models' measured
prediction costs are charged for their hybrid; setup/warmup are separate, so
this is not cold lifecycle or independently timed per-request inference.
376SAT outputs pass additive complete-assignment checks;11UNSAT formulas are
independently corroborated. No retraining, new confirmation or paper rewrite.
Next: design solver-cost-supervised joint heads with strong cheap/native-RLAF
controls, not a direct combination of existing checkpoints. Not launched.

Earlier method diagnostic: A29 tests five coloring/branching priorities plus the
actual inherited search on128reused+128new TRAINING graphs.61,440search calls
are not independent instances or full-cost measurements. A28's matched/residual
resident search stage is only0.6478%/0.2722%of the recorded full cost: deleting
search alone cannot reach the5%hurdle if other stages stay fixed. The new residual
degree rule reduces nodes8.8125→5.6719 but CPU time grows1.995→2.282us.
No current-small-graph policy training launched; the harder/general-CNF audit
and separately frozen development mechanism study are now recorded as A30.

Latest audited performance update: A28 omits unused SLS indices only in the guarded
nonlearned entry. Matched exact-resident full cost falls19.60%
(1.605263→1.290584ms; corrected lower gain0.229563ms), witness32 falls7.30%, and
the matched strongest-resident gate passes. Sparse-stress exact resident worsens
14.71%; only10/24 total gates pass. Waiting tails remain unresolved. A27 and A25's
earlier local gains and every historical failure stay separate. No promotion or
learned advantage is claimed.

Open-source, dependency-free research progress dashboard. MIT licensed.

[Live dashboard](https://asimfish.github.io/cap-sat-dashboard/)

The page separates experiment completion from resolution of research weaknesses.
It includes SAT/UNSAT views, comparator selection, per-seed progress, evidence
limitations, offline snapshots, and a stale-data indicator. Layout inspiration:
[PsiBot dashboard](https://asimfish.github.io/psibot-dashboard/); no code copied.

## Rebuild

```sh
python3 build.py --repo /path/to/cap-sat --out site
python3 -m http.server 8080 --directory site --bind 127.0.0.1
```

Open http://localhost:8080, or open `site/index.html` directly for the offline
snapshot. No Python packages or Node build are required. The public repository
does not include raw experiments: point `--repo` at an existing CAP-SAT checkout
containing `experiments/sprint_20260905`. A missing dataset displays zero rows,
not completed experiments.

`issues.json` is the human-maintained evidence ledger; `build.py` computes
E17/E19 n350 progress and PAR-2 contrasts from solver logs. Edit the ledger or
`template.html`, then rebuild; do not edit generated HTML. Every export records
the data fingerprint and timestamp. The browser checks `status.json` each minute;
an exporter must separately regenerate and publish it. Browser refresh alone
cannot read private experiment files. The displayed export freshness is not a
process heartbeat; last log timestamps are shown independently.

## Performance roadmap

`performance_plan.json` is the canonical, manually reviewed seven-gap roadmap.
Each gap separates observed progress, remaining limitations and source maturity.
Thirty-seven actions have dependencies, costs, acceptance and stopping rules. Their
reviewed status is distinct from automated experiment progress. E26 is the first
bounded new pilot; historical K%-phase results still require complete row-level
reconciliation before promotion to confirmed performance gains.
The roadmap's review date and SHA256 are separate from live export freshness.

`#action-A28`:128 new development graphs,7680full-cost cells and4260SAT witnesses
replayed. Both profiles use original exact without a prefix. Token parsing,
original clauses, recognition, search, all native/caller checks, fallback and
full lifecycle accounting stay unchanged; five controls receive the same change.
The restricted nonlearned binary does not expose the old general/SLS CLI.
Matched8/12 and stress2/12 gates pass; parse mechanisms10/10 and6/10. Resident
matched exact and witness32 improve19.60%/7.30%, but stress exact resident
2.059603→2.362667ms regresses14.71%. All tail events remain. A 2,560-call post-hoc
diagnostic does not reproduce the20ms tails; file open averages~300us, file close
~4us. It neither fixes nor identifies historical NFS/scheduler or process-exit
tail causes. Next: examine a fully charged input-transfer path to avoid repeated
file opens; do not hide serialization, transfer or validation costs. Not launched.
This is development, not confirmation, CAP learning or industrial advantage.

`#action-A27`:128 newly generated development graphs in two64-graph panels,
11520full-cost cells and5940SAT witnesses independently replayed. Baseline,
shared32 and shared1 are measured in the same binary. Only exact's prefix changes;
witness32 retains32 attempts, the1ms total graph deadline and full original-CNF
checks remain. Release and ASan/UBSan correctness pass. The sparse-stress resident
repair passes its3%/positive-interval gate; graph-search-stage mean falls86.70%,
which does not mean ExactSearch itself accelerated. Matched6/14 and stress9/14
gates pass. Matched n48 resident request+validation is17.949us slower than witness32;
matched cold mean regresses7.24%. Tails and all failures remain. Witness32 wins
already present in original exact are not attributed to shared1. No new confirmation,
production switch, learned advantage, or alteration of A26's eight-hour campaign.
Next: diagnose same-witness n48 validation, input parse and cold exit costs; do not
search more prefix counts or use timing noise to select a winner.

`#action-A25`:64fresh graphs,3840full-cost cells,3120SAT witnesses independently
replayed using the old checker. Both implementations call a reusable verified
request API, which checks every original clause and complete assignment before
returning SAT. Exact resident1.655→1.152ms(-30.38%), witness32 resident2.001→
1.492ms(-25.44%) and witness cold6.614→6.035ms(-8.74%) pass. Exact cold and random
cold gates fail; all10validation-stage mechanisms pass. The native binary stays
unchanged. No CNF preconversion, result cache or correctness-check removal.
Direct strongest resident mean favors exact22.75%, but the predeclared request
plus validation scale guard fails by5.819us at n64; cold strongest also fails.
Full-cost gains do not establish a complete strongest-baseline win, native-core
speedup, CAP learning, independent confirmation or production deployment.

`#action-A24` is a manually audited readout:64fresh graphs,3840full-cost cells
and3120SAT witnesses replayed. Exact resident2.480→2.404ms(-3.07%) passes its
prespecified gate; witness32 resident5.772→5.723ms fails the2%/interval/scale gate.
Exact and witness32 cold pass noninferiority only; stock/degree/random cold fail.
Compiled trusted Horn templates reduce witness completion~76%, without changing
search or removing original-CNF checks. All five strategies get both profiles;
fully charged warmups and tails are retained. A23's rejected reaper is not used.
Next: profile the Python full checker (~39%of candidate exact-resident cost),
retaining every original-clause check and charging conversion. Not launched.
No learned advantage, overall repair, production switch or new confirmation.

`#action-A23` is a manually audited readout, not an automatic collector:64fresh
graphs,3840full-cost measurements and3180SAT witnesses replayed. Exact cold
6.891→6.096ms passes its scoped gate; exact resident1.805→1.809ms is noninferior,
not a speedup. Five matched nonlearned controls get both close policies; startup,
thread creation/reaping, validation and resident warmup/lifetime costs count.
The rejected EOF prototype and failed overall reaper gates are retained. Separate
instrumented replay locates a new40ms close event in thread startup without GC
overlap, but does not prove a kernel/scheduler cause or fix historical tails.
A21/A22 gains use different cohorts and must not be added to A23's percentage.
No learning advantage, production switch or new confirmation is claimed.

`#recognition-repair` / A20 reports 64 fresh graphs, each in three equivalent
clause/literal orders, three repetitions, four arms and two lifecycles: 4,608
measurements, not 192 independent graphs. Canonical recognition and exact search
are unchanged; strict template-multiset matching restores the reordered path.
The six prespecified one-sided repair/noninferiority gates all pass. Costs include
startup, IPC, native/Python witness checks and exit; resident lifetimes and separate
warmup costs are allocated to 64 requests. Bootstrap resampling is by graph with
Bonferroni adjustment across the six one-sided gates. Canonical noninferiority is
not strict improvement, and witness/degree comparisons are descriptive here.
The exporter binds raw, charged, frozen and result digests, checks aggregate
denominators/finite values and recomputes gates; it is not a substitute for the
experiment's full witness/account replay. Public data are allowlisted aggregates.
E65's stronger confirmation remains failed; neither learned advantage nor general
SAT/industrial acceleration or a production route switch is claimed. Seven
scientific gaps remain four partial, three open and zero fully solved.
Edit this JSON, rebuild and publish; the browser filters partial progress versus
unresolved gaps and links each gap to its proposed experiments. The private
discussion attachment, reviewer identities and confidential correspondence are
not published. Publishing a plan does not authorize or launch an experiment.

The E26 card reads allowlisted counts from its frozen design and progress file;
it shows no partial-result ranking. Only a complete 24-row / 144-cell readout
with the matching frozen-file hash supplies the six-arm results and four
simultaneous contrast intervals. The exporter checks aggregate shape, ranges
and identity, not every raw solver record; terminal scientific interpretation
remains a separate review. A recent progress update is not proof a process is
still alive. SAT witnesses are checked and UNSAT outcomes independently
corroborated, not proof-certified. Single-checkpoint, small-sample, shared-host
pilot results cannot by themselves establish an advantage or justify rewriting
the paper. Raw vectors, local paths, host details and logs are never exported.

E27 is displayed separately as24 development instances and (only after frozen
selection)48 fresh confirmation instances at n325/n500. It distinguishes a cheap
witness-first repair/CDCL fallback engineering benefit from CAP's incremental
learning benefit over matched cheap repairs. Passing one is not passing the
other. Per-scale tables retain all denominators; displayed simultaneous contrast
intervals always refer to the overall prespecified comparison, not the selected
scale. Only complete stage readouts are ranked; all published fields are allowlisted.

E28 records a rejected short-rollout utility model, with no confirmation panel.
E29 separately records direct capped-solve policy optimization:18432 training
cells,96 development formulas x3 search-seed repetitions, and conditional192
fresh confirmation formulas. Training fitness is penalized flips, not seconds.
Full-cost tables show repeated search trials, not inflated independent-instance
counts. Published E29 data exclude policy coefficients and private CPU affinity.
New functionality remains linked to A7; neither a completed run nor a lower
training fitness automatically upgrades a scientific gap to solved.

E30 (#cost-aware/A8) trains on0.25s charged wall-time rather than flips, with
optional exact-zero feature masks.9216 cells on64 independent training formulas
finished; both winners are exactly the existing hand rule and skip CAP. A hashed
preflight/audit receipt records the explicit protocol deviation: stop before
planned development, with neither development nor test generated. This is NOT
a statistically failed development gate or a confirmed performance result.
Training fitness uses seconds, distinct from E29's flip units and5s evaluation.
A5 separately records E31's small-graph encoding correctness preparation; no
historical dataset migration is claimed.

E32 (#shared-gpu/A9) adds actual disjoint-graph batching and three CSM-only
adaptation seeds on shared GPUs 3/2/5. All three completed 120 epochs on 128 new
training graphs with 32 validation graphs. The matched, GPU-resident 8-graph
forward/backward benchmark improved throughput 7.51x; this is neither the full
old joint training pipeline nor single-instance SAT latency. Per-formula feature
normalization and loss weighting preserve checkpoint semantics. The collector
rechecks both checkpoint hashes and all three supervisor exit codes before
declaring terminal completion. Displayed utilization samples describe whole
cards while our tasks were alive, including other jobs; they are not causal
measurements of our own utilization. Peak memory is PyTorch allocated memory,
not total process/CUDA-context memory. A separately frozen resource replay runs
the unchanged workload and samples total GPU process memory by owned PID; its
sampled peaks are shown separately and do not backfill the missing original
measurement or count as new independent scientific seeds. Public exports omit process IDs, device
UUIDs, unrelated process inventories, private paths and model weights. A lower
validation CSM loss does not upgrade any performance gap. Subsequent E33 solver
development is separate and did not pass; it does not rewrite E32 training results.

E33/E34 (#structured/A10/A11) each completed a DIFFERENT 64-graph development
panel, three search repetitions and eight arms (1536 cells each). E33 adds feasible
graph decoding with guarded default fallback; E34 trains a new 113-parameter
policy directly on independent-set size, not the original CAP objective. Original
CAP, adapted/new seeds, untrained models and cheap controls remain separate.
Both failed the prespecified full-cost gates and generated no confirmation set.
The collector requires a matching terminal audit/readout/selection before ranking,
and exports only fixed aggregate fields. Tables use milliseconds and original
graph-vertex counts, not CNF auxiliary-variable counts. SAT witnesses are checked;
global UNSAT is independently corroborated, not proof-certified. Assumption-UNSAT
cannot be reported as global UNSAT. Shared-host long tails and component-charged
resident timing remain limitations. A12 is now implemented as the separate E35
experiment described below; E33/E34 frozen results are not rescored. No scientific gap is promoted
by validation rewards, new-model inference savings or selected point estimates.

E35 (#resident/A12) implements native113-parameter inference and a private resident
child, parsing each request once and recreating the solver. A paired conversion
benchmark on8 existing validation graphs x10 repeats compares old subprocess,
new cold subprocess and new resident, for BOTH default and the same seed42 policy.
Tiny-policy complete median11.212→2.203ms and mean29.708→9.459ms pass the20%
conversion gate; cold/load/warmup separate, numerical parity2e-6, no original-CAP
compression claim. Eight new resident development arms include static degree
WITHOUT a network as well as matched untrained, dynamic/random/default controls
and all three tiny seeds. On64 new graphs x3 repeats, all arms192/192 solved;
tiny PAR2 means8.171/9.285/9.589ms vs dynamic degree7.374ms. Both prespecified
development gates fail, no confirmation generated. Shared-host tails remain;
CPU execution time was not recorded. The collector requires matching benchmark
and development audits and never combines their sample denominators or claims.
A13 is a planned deployment-aligned learning-signal investigation, not executed.

Only complete, error-free 300-row seeds enter metrics. Invalid indices,
non-finite times, arm errors, conflicting duplicate rows, differing CNF paths,
and contradictory SAT/UNSAT outcomes are rejected or excluded. Identity checking
is path-based here; this dashboard is not a substitute for the manuscript's
checkpoint/manifest/CNF-byte provenance audit. Confidence intervals resample
seeds only on the fixed test panel. Cross-host and contention effects remain.

Native RLAF/NeuroBack progress reads an allowlist of fields from
`experiments/native_baselines_20260910/SUPERVISION.json`. Process IDs, commands,
absolute paths and machine resource inventories are not exported. A heartbeat
older than five minutes is shown as unknown runtime status. Training completion
and formal comparative evaluation remain separate milestones.
For terminal training, the exporter rechecks receipt exits, accepted artifact
hashes and completed iteration/epoch coverage. Verified terminal records remain
completed after the monitor exits; altered/missing artifacts fail verification.

Only aggregate data and manually reviewed issue descriptions are published.
No raw CNF, model, credential, hostname, or absolute experiment path is exported.
E36 `#decoder-utility` separates the old32-graph proxy/deployment diagnosis from
new64-graph selected validation. Three-seed600-step training is complete, but
the across-seed gate failed: no development, test or solver-speed claim. The
collector requires matching audited readout identities and consistent per-scale
counts; the UI preserves the seed43 step0 checkpoint and n48 regression. E37/A14
has a separate `#stability` panel: a frozen four-candidate/three-seed validation
grid selects the pairwise small policy, but neither native development nor
independent confirmation has run. Positive training selection cannot upgrade
the seven scientific gaps. E38/E39 now add a native/full-cost development panel;
A15 records the failed native and size-adaptive learning gates and the next
instance-level online-selection frontier. Labels, model
choices, checkpoints and process identities stay private.
`evidence.html` is a dated, offline evidence summary generated using the
render-html skill; it is an archival snapshot, not the live exporter.

## Publish / automatic updates

`publish.py --repo /path/to/cap-sat --checkout /path/to/dashboard-clone`
rebuilds, synchronizes an explicit file allowlist, and pushes only when source
content changes or the last exported snapshot is at least one hour old.
It requires an authenticated Git checkout of this public dashboard repository.
Add `--stage-only` to stop after staging the allowlist for review, without a
commit or push. Review the staged diff before manually publishing it.
`watch.py` runs this check every 15 minutes with a single-instance file lock and
a 180-second per-export deadline. It can run in tmux; it does not survive host
reboots unless a host scheduler starts it again. Run collector regression checks
with `python3 -m unittest discover -s . -p 'test_build.py'` in this directory.
Use a scheduler with a non-overlap lock, for example:

```sh
flock -n /tmp/capsat-dashboard.lock python3 publish.py --repo /path/to/cap-sat --checkout /path/to/dashboard-clone
```

Do not serve the CAP-SAT experiment root. GitHub Pages serves `site/` through
the included Actions workflow. The collector is read-only with respect to the
experiment repository and never launches solver or training jobs.

## A32 local-action / budget calibration

The top A32 panel tracks 24 new training-calibration CNFs, 6 generation streams,
and 2,448 feedback cells (not independent samples). It compares full/random8/
random16/CAP-ranked16 perturbations around the same strong RLAF anchor at
0.75/3/6 seconds, with unchanged-anchor and minimal-degree controls. Two shared
GPUs only prepare predictions; up to 24 CPU workers produce solver feedback.
The collector redacts process identities and paths, marks stale heartbeats, and
never interprets queue completion as learned or end-to-end advantage. Audited
scientific conclusions belong in the A32 plan card. This pilot does not train a
new model, auto-expand the queue, or modify paper conclusions.
