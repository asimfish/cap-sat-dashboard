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

Only complete, error-free 300-row seeds enter metrics. Invalid indices,
non-finite times, arm errors, conflicting duplicate rows, differing CNF paths,
and contradictory SAT/UNSAT outcomes are rejected or excluded. Identity checking
is path-based here; this dashboard is not a substitute for the manuscript's
checkpoint/manifest/CNF-byte provenance audit. Confidence intervals resample
seeds only on the fixed test panel. Cross-host and contention effects remain.

Only aggregate data and manually reviewed issue descriptions are published.
No raw CNF, model, credential, hostname, or absolute experiment path is exported.
`evidence.html` is a dated, offline evidence summary generated using the
render-html skill; it is an archival snapshot, not the live exporter.

## Publish / automatic updates

`publish.py --repo /path/to/cap-sat --checkout /path/to/dashboard-clone`
rebuilds, synchronizes an explicit file allowlist, and pushes only when source
content changes or the last exported snapshot is at least one hour old.
It requires an authenticated Git checkout of this public dashboard repository.
Use a scheduler with a non-overlap lock, for example:

```sh
flock -n /tmp/capsat-dashboard.lock python3 publish.py --repo /path/to/cap-sat --checkout /path/to/dashboard-clone
```

Do not serve the CAP-SAT experiment root. GitHub Pages serves `site/` through
the included Actions workflow. The collector is read-only with respect to the
experiment repository and never launches solver or training jobs.
