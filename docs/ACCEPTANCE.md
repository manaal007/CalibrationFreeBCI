# Current executed acceptance record — external release v4

Version **0.3.0** has completed the authorized frozen PhysioNet evaluation.
The run is **provisional**: prior participant exposure remains unconfirmed.
Approval to run was received; no repeat approval is required for reproducing it.

| Current check | Executed outcome |
|---|---|
| Package installation | PASS: editable 0.3.0 installation and wheel build; wheel installed into separately created validation environment; installed modules match source hashes |
| Automated tests | PASS: 20 tests in project environment and clean installed-wheel environment |
| Raw/source/external pipeline | PASS: previously verified source workflow; real PhysioNet S001–S005 inference on 225 trials with two frozen models |
| Model integrity | PASS: original selected model and comparison models retain their frozen checksums; no refitting or threshold adjustment |
| External metrics | PASS: all 450 prediction rows reproduce 10 participant/model metric rows |
| Repeatability | PASS: predictions.csv and participant_metrics.csv are byte-identical across three runs, including installed-wheel execution outside the source tree |
| Reports and figures | PASS: participant tables, bootstrap intervals, paired comparison, PNG/SVG class and participant figures, Markdown report |
| Publication file checks | PASS: Git-visible tree checked for raw EEG, local environments, model/archive files and files over 50 MiB; none found |
| GitHub CI | IMPLEMENTED: Windows/Python 3.12 install and test workflow; first remote execution remains pending a push |
| Independent scientific success | NOT ESTABLISHED: cohort exposure history unconfirmed; accuracy interval includes chance and improvement interval includes zero |

Primary external outcome: aligned CSP participant-mean balanced accuracy
**0.6044** (95% participant-bootstrap interval **0.4332–0.7658**), compared with
**0.5614** (0.4940–0.6289) for RMS/CSP. Paired difference **0.0430**
(−0.0750 to 0.1634). Two aligned participants performed below chance and were
retained unchanged. All models produced both classes for each participant,
without forcing balanced predictions.

Evidence:

- [Frozen external report](../results/physionet_external_v4/REPORT.md)
- [Release verification](../results/external_release_v4/release_verification.json)
- [Publication file check](../results/publication_check.json)
- [Publishing instructions](PUBLISHING.md)

The original archive and earlier run directories remain preserved. This cohort
is now exposed; further tuning requires a new independent cohort for a fresh
confirmation. No GitHub remote was supplied and no push was performed.

## Historical v3 acceptance record — before the authorized PhysioNet run

The following records what was completed at the previous checkpoint. Its pending
evaluation statements are historical and are superseded by the v4 record above.

Environment: Windows, PyCharm-configured Python 3.12.10. Active branch:
`research/rigorous-pipeline`. These statuses describe executed checks, not
assumptions about historical scripts or independent scientific success.

| Acceptance item | Status and actual evidence |
|---|---|
| Dependencies install | PASS: editable installation; built wheel; fresh virtual environment installed 86 pinned cached packages offline; `pip check` passed |
| Raw EEG loading | PASS: all 18 BNCI files / 108 runs; real Cho participant 1; 15 real PhysioNet EDFs for S001–S005 |
| Preprocessing | PASS: units, 22-electrode ordering, reference and shape checks; source and target fixtures; real loader runs |
| Timing and labels | PASS within stated evidence: MATLAB indices match installed converter; direct filtered sample slices match epochs; original specification supports two-second cue delay. No local GDF-to-MAT comparison was possible |
| Artifact handling | PASS: supplied flags retained and exported; metadata rejection option tested on MATLAB fixtures; primary runs intentionally retain flagged trials |
| Model training | PASS: original five methods, three-candidate nested filter-bank grid, fixed aligned CSP comparison; CPU only |
| Source validation | PASS: nine-participant LOSO and 3×3 nested participant folds; no target labels in selection |
| Serialization | PASS: actual model checksums, reload predictions, corruption tests and clean-environment inference |
| Cross-dataset inference | PASS technically: frozen source models ran on real previously examined Cho participant 1; synthetic PhysioNet inference passed. Real PhysioNet scoring is pending approval |
| Metric reproduction | PASS: all 11,368 prediction rows in the final workflow independently reproduce saved participant metrics |
| Automated tests | PASS: 18 tests in project environment and fresh installed-wheel environment |
| Figures and reports | PASS: source PNG/SVG comparisons, external confusion plots, paired participant intervals, Markdown reports; actual figures inspected |
| Documentation | PASS: README commands, manuscript, abstract, application description and confirmation protocol updated to distinguish exploratory and independent evidence |
| Clean setup | PASS: fresh venv, offline cached-wheel dependency installation, installed package imported outside repository, all 18 tests passed; real exploratory inference reproduced |
| Repeatability | PASS: repeated alignment source experiment has zero score difference across 5,184 prediction rows; three real external runs, including fresh environment, produce byte-identical prediction CSVs |
| Independent scientific confirmation | PENDING: explicit owner approval and confirmation of previously unused PhysioNet participants are required |

Primary verified evidence: [verification.json](../results/final_v3/validated/verification.json),
[alignment report](../results/final_v3/validated/alignment_report/REPORT.md),
[nested report](../results/final_v3/validated/nested_report/REPORT.md),
and the logs and loading/timing evidence in
[verification artifacts](../results/final_v3/validated/verification/).
The [final distribution check](../results/final_v3/validated/verification/final_distribution_check.json)
also verifies version 0.2.1, installed-module hashes against source, the final
18-test suite and byte-identical real exploratory predictions.

The complete command actually executed was:

```powershell
.\.venv\Scripts\python.exe -m calibrationfreebci workflow --out runs/verified_workflow_v3
```

The aligned source model reached mean LOSO balanced accuracy 0.7334 versus
0.6964 for matched RMS/CSP. The descriptive paired difference was 0.0370
(participant bootstrap interval 0.0085–0.0640); this does not correct repeated
development or model selection. Exploratory Cho participant 1 balanced accuracy
was 0.6950 versus 0.5350, with aligned left/right recall 0.58/0.81. These results
use an unlabeled target batch and are not independent population evidence.

The corrected nested grid was a negative result: 0.6817 balanced accuracy for
nested selection versus 0.7029 for fixed RMS/CSP on identical outer folds.
Earlier float32 bank runs were superseded after identifying numerical rank
instability; they were retained rather than overwritten.

The original archive checksum is unchanged. The ZIP read-only attribute and
checksum provide preservation and tamper detection, not hardware immutability.
No final confirmatory predictions were generated for PhysioNet. No dataset
downloads or GPU training were performed. The separate dependency-resolution
dry run succeeded but is not the basis for the clean-install PASS: actual offline
installation and execution provide that evidence.
