# CalibrationFreeBCI

[![Tests](https://github.com/manaal007/CalibrationFreeBCI/actions/workflows/tests.yml/badge.svg)](https://github.com/manaal007/CalibrationFreeBCI/actions/workflows/tests.yml)

Reproducible CPU research on left-versus-right motor imagery EEG decoding across participants and datasets, without supervised target calibration. Works with local data on Windows/PyCharm.

**Current external evidence:** In a frozen, provisional BNCI-to-PhysioNet evaluation on five participants, aligned CSP achieved **60.4% participant-mean balanced accuracy** (95% participant-bootstrap interval **43.3–76.6%**) versus **56.1%** for RMS/CSP. The paired improvement interval includes zero; reliable population-level improvement is **not established**. Prior participant exposure was not confirmed, so these results are not called confirmatory.

Source LOSO balanced accuracy was 73.3% for aligned CSP versus 69.6% for RMS/CSP. A previously examined Cho participant yielded exploratory balanced accuracy 69.5%. Alignment uses an **unlabeled participant batch**, not strict single-trial inference. See the [complete external report](results/physionet_external_v4/REPORT.md), including below-chance participants. No thresholds or models were tuned on PhysioNet outcomes.

The original experiments are preserved. A corrected, supported package replaces the numbered scripts as the execution path. Historical scripts remain evidence and are not all supported entry points.

## Install and run

Use Python 3.12. From the repository root in PowerShell:

```powershell
# For a fresh clone; skip creation if PyCharm already configured .venv
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m calibrationfreebci workflow --out runs/my_complete_run
```

The workflow runs nested source development, freezes models, verifies saved metrics, performs explicitly exploratory Cho2017 inference, generates figures/reports, then runs the fixed alignment comparison and its exploratory evaluation. It does not run final PhysioNet confirmation. Output directories must be new. Defaults are packaged with the installed module; `--config configs/nested.json` supplies an explicit configuration. `main.py`, `python -m calibrationfreebci`, and the installed `bci-research` command use the same CLI.

Both sessions of all nine BNCI participants and Cho participant 1 must be cached locally. There are **no automatic dataset downloads**. Discovery checks the existing Hugging Face/MNE caches. For another layout, use `--data-root PATH` for BNCI and `--cho-root PATH` for the folder containing `s01.mat`. Missing files fail with their exact names. Models and local runs are excluded from Git; curated result tables and figures live under `results/`.

## Commands

```powershell
# Existing five-method source LOSO comparison and final model
.\.venv\Scripts\python.exe -m calibrationfreebci source --out runs/source_comparison

# Smaller real-data smoke run
.\.venv\Scripts\python.exe -m calibrationfreebci source --subjects 1 2 3 --out runs/source_smoke

# Nested, participant-separated model/component selection
.\.venv\Scripts\python.exe -m calibrationfreebci nested --config configs/nested.json --out runs/nested

# Fixed label-free alignment versus RMS/CSP: source LOSO and frozen models
.\.venv\Scripts\python.exe -m calibrationfreebci alignment --out runs/alignment

# Exploratory cross-dataset inference, reload checks, saved metrics and figure
.\.venv\Scripts\python.exe -m calibrationfreebci cross-smoke --source runs/alignment --out runs/cho_smoke

# Recompute saved metrics without training
.\.venv\Scripts\python.exe -m calibrationfreebci verify --run runs/cho_smoke

# Source/external report and figures from saved outputs
.\.venv\Scripts\python.exe -m calibrationfreebci research-report --source runs/alignment --external runs/cho_smoke --out runs/report

# Real target loading checks WITHOUT predictions or decoding metrics
.\.venv\Scripts\python.exe -m calibrationfreebci target-qc --out runs/physionet_loading.json

# Raw sample-coordinate evidence and installed converter checks
.\.venv\Scripts\python.exe -m calibrationfreebci verify-timing --out runs/timing.json
```

The `source` command produces the original five-method development comparison. Before using its output for inference, run `freeze-comparisons --source runs/source_comparison`. The `nested`, `alignment`, and `workflow` commands freeze their comparisons automatically.

## Event timing and preprocessing

The original competition specification distinguishes trial start (code 768) from imagery cue onset (codes 769–772), two seconds later. The cached BNCI MATLAB `trial` field is one-based. The installed MOABB converter places class triggers at `trial - 1`; its dataset interval is [2,6] seconds. The implemented window is [2.5,4.5] seconds after trial start, or [0.5,2.5] after the cue. Verification compares all 108 cached runs and directly checks filtered raw slices against epochs. A local GDF file was unavailable, so no direct GDF-versus-MAT timestamp comparison is claimed.

Use 22 explicitly ordered EEG electrodes, volts, their average reference, 250 Hz, and 501-sample epochs. Filter continuous source runs before epoching. Cho uses original seven-second trial blocks, filtered before resampling 512 to 250 Hz; it never filters the legacy zero-buffered concatenation. PhysioNet uses continuous EDF runs 4/8/12, T1=left and T2=right imagery, native 160 Hz filtering and rational resampling. Recording systems and finite-segment filter boundaries still differ.

Source artifact flags are retained in primary analyses and reported in metadata. `load_bnci(..., artifact_policy='reject_flagged')` implements tested metadata-based rejection; the nested config can select it before a run. Primary comparisons retain flags so cohorts match. Historical clean-flag sensitivity is saved separately. No performance-dependent participant rejection is used.

Average reference removes one spatial degree of freedom. The filter-bank implementation retains float64 data, reapplies the common reference and specifies CSP rank explicitly. Earlier development runs that compressed these signals to float32 are superseded because automatic rank estimation could fit a numerical null component. Their outputs remain on disk as evidence.

## Models and scientific protocol

The five-method comparison includes CSP/LDA, OAS/MDM, RMS/CSP/shrinkage LDA, trace-normalized MDM, and trace-normalized tangent-space/shrinkage LDA. The small nested grid compares RMS/CSP with four components to four-band CSP using two or four components per band. Bands are 8–12, 12–18, 18–24 and 24–30 Hz. The three outer and three inner folds split participants, not trials. The final candidate is selected on source participants only. Do not compare nested three-fold estimates directly to nine-fold LOSO as if their training sets were identical.

The separate fixed alignment method normalizes trial RMS, subtracts temporal means, estimates the participant's mean trace-normalized covariance, adds 1% diagonal loading, and whitens that participant's trials before RMS/CSP/shrinkage LDA. Source participants are aligned separately. At inference, each target participant is aligned from their **unlabeled batch**. This is unsupervised participant adaptation; batch composition can change predictions, and it is not strict inductive single-trial inference. Labels are never supplied to the alignment function. This is a regularized variant motivated by Euclidean alignment, not an exact replication of its original implementation.

All scores are signed toward right imagery; thresholds stay at zero. There is no target-label threshold tuning and no forced balancing of predictions. Reports include balanced accuracy, ROC-AUC, class recalls, predicted-right fraction, trial identities and participant-level results. Confidence intervals resample participants, not individual trials. Source model selection and repeated development introduce selection optimism; exploratory intervals do not correct it. A single external participant cannot support population-level confidence intervals.

## Frozen PhysioNet evaluation

The owner authorized evaluation, and the frozen selected model plus its matched RMS/CSP baseline were run on all five cached participants (225 trials). Metric verification and repeated inference passed. The run is explicitly **provisional** because prior participant exposure remains unconfirmed. The confirmatory gate still requires owner confirmation of previously unused participants; evaluation approval does not establish that history.

To reproduce the authorized provisional run with newly created output paths:

```powershell
.\.venv\Scripts\python.exe -m calibrationfreebci external --source runs/my_complete_run/alignment_source --subjects 1 2 3 4 5 --out runs/physionet_repeat --evaluation-kind provisional --exposure-record configs/physionet_exposure.provisional.json --approved-external-evaluation
.\.venv\Scripts\python.exe -m calibrationfreebci verify --run runs/physionet_repeat
```

This produces predictions, participant metrics, confidence intervals, paired differences, PNG/SVG figures and a Markdown report automatically. To regenerate derived tables and figures, use `external-report --run runs/physionet_repeat`. Model weights and thresholds remain frozen. Below-chance participants are retained, without label flipping.

Confirmatory mode remains the default and requires an owner-confirmed exposure record based on `configs/physionet_exposure.template.json`. The template itself is not approval. S001–S005 have now been evaluated, so do not tune on their outcomes and subsequently call a rerun fresh confirmation. Further method development requires a separately held-out cohort for a new confirmation.

Five participants provide wide uncertainty. The existing PhysioNet results do not establish robust population generalization. See [publication instructions](docs/PUBLISHING.md) before pushing the repository.

## Outputs, verification and portfolio

Each experiment saves its protocol before fitting, raw-file hashes, participant metrics, trial predictions, summaries, selected model and frozen comparison manifest. Reports include PNG/SVG scientific figures and paired participant differences. Serialized joblib models should be loaded only from trusted project outputs.

- [Manuscript](docs/MANUSCRIPT.md), [abstract](docs/ABSTRACT.md), [application description](docs/APPLICATION_DESCRIPTION.md)
- [Final acceptance record](docs/ACCEPTANCE.md) distinguishes executed checks from pending confirmation
- [Verified source/alignment and exploratory transfer results](results/final_v3/validated/alignment_report/REPORT.md)
- [Corrected nested comparison, including negative findings](results/final_v3/validated/nested_report/REPORT.md)
- [Reproduction checks](results/final_v3/validated/verification.json)
- [Scientific weaknesses and proposed next experiment](docs/RESEARCH_REVIEW.md)
- [Independent protocol](docs/CONFIRMATORY_PROTOCOL.md)
- [Original preservation and checksum](PRESERVATION.md)

Tests cover raw MATLAB fixtures, artifact policies, timing, channel ordering, unit scale, nested participant separation, class mapping, score orientation, gain invariance, model serialization, participant-batch isolation, corrupted metrics/models, and synthetic EDF inference. Real data runs supplement these tests. `scripts/validate_clean_setup.py` tests a fresh virtual environment using locally cached wheels without network downloads; missing wheels produce an explicit failure. A wheel build and isolated package tests check that the distribution contains its default configuration.

## References

- [Original BCI Competition IV dataset 2a specification](https://www.bbci.de/competition/iv/desc_2a.pdf)
- [BNCI2014-001 loader and protocol](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html)
- [PhysioNet EEG Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [He and Wu: Euclidean-space data alignment](https://arxiv.org/abs/1808.05464)

Research software. The completed provisional evaluation does not establish reliable independent cross-dataset performance.
