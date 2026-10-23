# Frozen external evaluation protocol and status

The owner subsequently authorized evaluation. The source-selected aligned CSP
and frozen RMS/CSP comparator were evaluated on S001–S005 in
`results/physionet_external_v4`. Prior participant exposure was not established,
so the completed run is **provisional**, not confirmatory. Model weights and
thresholds were unchanged. The primary mean balanced accuracy was 0.6044 versus
0.5614; the paired interval included zero. This cohort is now exposed.

The prospective requirements below are retained to distinguish the intended
confirmation protocol from the actually achieved evidential status.

Freeze the source run's protocol, model checksum and selection before evaluating targets.
Primary endpoint: participant-mean balanced accuracy; secondary endpoints: ROC-AUC,
left/right recall and predicted-right fraction. Use participant bootstrap intervals,
never trial bootstrap intervals to imply population certainty.

Candidate independent dataset: PhysioNet EEG Motor Movement/Imagery 1.0.0, runs 4,
8 and 12 only (imagined left/right fist). Five participants S001–S005 are cached.
Do not infer their untouched status from the absence of references in the current
repository. Obtain the owner's confirmation of prior use before treating results
as confirmatory. Five participants provide limited precision and no broad population claim.
Larger downloads require separate approval.

T1 = left, T2 = right. Use only the 22 BNCI electrodes, fixed ordering, MNE volts,
average reference, continuous-run 8–30 Hz filtering, rational resampling 160→250 Hz,
and inclusive 0.5–2.5 s post-task-onset windows. No supervised target fitting,
target-label threshold adjustment or participant exclusion based on performance.
The separate aligned protocol below permits label-free participant-batch alignment.
Fail on missing channels, incomplete segments or inconsistent sampling frequency.

Compare the selected method to fixed CSP/LDA and OAS/MDM baselines trained on
the same corrected source data. Freeze all comparison models before access to
external predictions. Report paired participant differences and confidence intervals.
Do not proceed with further tuning after seeing final outcomes; any subsequent
changes need a fresh independent cohort.

The fixed alignment development comparison selects between RMS/CSP and
regularized participant-aligned RMS/CSP using matched source LOSO balanced
accuracy. Its frozen manifest contains both models. This is an additional
explicit protocol: participant-aligned inference estimates an unsupervised
reference separately within each target participant's batch. It uses no target
labels but is not single-trial inductive inference. For the final aligned run,
use the source directory containing these frozen comparison models, report both
methods, and identify the primary selected model in advance. Historical
CSP/LDA and MDM exploratory results remain separate comparisons unless their
models are explicitly included in the frozen final manifest before evaluation.

The evaluator also requires `--exposure-record` with owner-confirmed cohort
history for confirmatory mode. The template has `owner_confirmed=false` and does
not authorize confirmation. The authorized provisional run records unknown
history without claiming confirmation, using a separate approval record.

Cho2017 is exploratory only: previous participants have already informed development.
Existing historical results used different epoch timing and must not be presented as
a controlled before/after test of normalization alone.

References: https://physionet.org/content/eegmmidb/1.0.0/ and
https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html
