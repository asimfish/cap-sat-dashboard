# CAP-SAT Rebuttal Dashboard

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
Eight actions have dependencies, costs, acceptance and stopping rules. Their
reviewed status is distinct from automated experiment progress. E26 is the first
bounded new pilot; historical K%-phase results still require complete row-level
reconciliation before promotion to confirmed performance gains.
The roadmap's review date and SHA256 are separate from live export freshness.
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
`evidence.html` is a dated, offline evidence summary generated using the
render-html skill; it is an archival snapshot, not the live exporter.

## Publish / automatic updates

`publish.py --repo /path/to/cap-sat --checkout /path/to/dashboard-clone`
rebuilds, synchronizes an explicit file allowlist, and pushes only when source
content changes or the last exported snapshot is at least one hour old.
It requires an authenticated Git checkout of this public dashboard repository.
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
